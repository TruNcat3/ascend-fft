// CPU-only ACL stub tests exercise host guards, not device arithmetic.
#include "butterfly/plan.hpp"
#include "butterfly/reference.hpp"
#include <cassert>
#include <cmath>
#include <limits>
#include <fstream>
#include <functional>
#include <stdexcept>

static std::string read(const char* path) {
    std::ifstream f(path);
    return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

static std::string replaced(std::string text, const std::string& from, const std::string& to) {
    size_t at = text.find(from);
    assert(at != std::string::npos);
    text.replace(at, from.size(), to);
    return text;
}

static void write(const std::string& path, const std::string& text) {
    std::ofstream f(path);
    f << text;
    assert(f.good());
}

static bool throws(const std::function<void()>& operation) {
    try { operation(); } catch (const std::exception&) { return true; }
    return false;
}

int main(int argc, char** argv) {
    assert(argc == 2);
    const std::string temporary = argv[1];
    const auto hardwareText = read("config/ascend910_93_profile.json");
    const auto spaceText = read("config/butterfly_space.json");
    const auto hardwarePath = temporary + "/hardware.json";
    const auto spacePath = temporary + "/space.json";
    assert(throws([] { bfly::Hardware::load("/nonexistent-hardware-config"); }));
    assert(throws([] { bfly::DesignSpace::load("/nonexistent-space-config"); }));
    for (const auto& bad : {std::string("{}"), hardwareText + " trailing",
            replaced(hardwareText, "\"vector_core_num\": 48", "\"vector_core_num\": 0"),
            replaced(hardwareText, "\"ub_bytes_per_core\": 196608", "\"ub_bytes_per_core\": \"196608\""),
            replaced(hardwareText, "\"aicore_num\": 24", "\"aicore_num\": 24.5"),
            replaced(hardwareText, "\"schema_version\": 1", "\"schema_version\": 2")}) {
        write(hardwarePath, bad);
        assert(throws([&] { bfly::Hardware::load(hardwarePath); }));
        bfly::Context invalid;
        assert(invalid.init(hardwarePath, "config/butterfly_space.json", "unused.o") == -4);
    }
    for (const auto& bad : {std::string("{"), std::string("{}"), spaceText + " trailing",
            replaced(spaceText, "\"ud_core\": [1, 8, 24, 48]", "\"ud_core\": []"),
            replaced(spaceText, "\"radix\": [2, 4, 8]", "\"radix\": [\"2\"]"),
            replaced(spaceText, "\"ts\": [1, 0]", "\"ts\": [2]"),
            replaced(spaceText, "\"out\": [\"interleaved\", \"planar\"]", "\"out\": [\"bogus\"]")}) {
        write(spacePath, bad);
        assert(throws([&] { bfly::DesignSpace::load(spacePath); }));
        bfly::Context invalid;
        assert(invalid.init("config/ascend910_93_profile.json", spacePath, "unused.o") == -4);
    }
    assert(acl_test::streamsCreated == 0);
    float reference[] = {0.f, 0.f};
    float output[] = {1.f, 0.f};
    assert(bfly::maxRelScaled(output, reference, 2) == 1.0);
    output[0] = 1e-6f;
    assert(bfly::maxRelScaled(output, reference, 2) > 0);
    reference[0] = 1e-8f;
    assert(bfly::maxRelScaled(output, reference, 2) < 1e-4);
    for (float v : {std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::infinity(),
                    -std::numeric_limits<float>::infinity()}) {
        output[0] = v;
        assert(std::isinf(bfly::maxRelScaled(output, reference, 2)));
        output[0] = 0;
        reference[0] = v;
        assert(std::isinf(bfly::maxRelScaled(output, reference, 2)));
        reference[0] = 0;
    }
    assert(std::isinf(bfly::maxRelScaled(nullptr, reference, 2)));
    assert(bfly::maxRelScaled(nullptr, nullptr, 0) == 0);
    reference[0] = 100.f;
    output[0] = 101.f;
    assert(bfly::maxRelScaled(output, reference, 2) == .01);

    bfly::Context context;
    assert(context.init("config/ascend910_93_profile.json",
                        "config/butterfly_space.json", "unused.o") == 0);
    assert(acl_test::streamsCreated == 1);
    assert(context.init("config/ascend910_93_profile.json",
                        "config/butterfly_space.json", "unused.o") == -9);
    assert(acl_test::streamsCreated == 1);
    auto candidates = context.enumerate(64, 1);
    std::unique_ptr<bfly::Plan> plan;
    for (const auto& candidate : candidates) {
        if (candidate.state != bfly::State::Infeasible &&
            bfly::DesignSpace::pointSize(candidate.p.radix, candidate.p.fusionLevel) == 2) {
            plan = context.makePlan(candidate);
            break;
        }
    }
    assert(plan);
    const auto canonical = plan->candidate();
    const std::vector<std::function<void(bfly::Candidate&)>> mutations = {
        [](auto& c) { c.p.bitReverseInput = false; },
        [](auto& c) { c.p.fusionLevel = 0; },
        [](auto& c) { c.p.coefficientResidency = false; },
        [](auto& c) { c.p.radix = 4; },
        [](auto& c) { c.l.contiguous = false; },
        [](auto& c) { c.l.in = bfly::Layout::Planar; },
        [](auto& c) { c.l.out = bfly::Layout::Planar; },
        [](auto& c) { c.a.td = 2; }, [](auto& c) { c.a.tb = 2; },
        [](auto& c) { c.a.ub = 2; }, [](auto& c) { c.a.us = 2; },
        [](auto& c) { c.a.ts = 0; }, [](auto& c) { c.a.level = 1; },
        [](auto& c) { c.a.udCore = 0; },
        [](auto& c) { c.a.localExchange = "register"; },
        [](auto& c) { c.a.localExchange = "shuffle"; },
        [](auto& c) { c.f.butterflyCount = 2; },
        [](auto& c) { c.f.stageCount = 2; },
        [](auto& c) { c.f.dftMatrix = true; },
        [](auto& c) { c.state = bfly::State::Infeasible; }
    };
    int loads = acl_test::binaryLoads;
    for (const auto& mutate : mutations) {
        auto candidate = canonical;
        mutate(candidate);
        assert(!context.makePlan(candidate));
    }
    assert(acl_test::binaryLoads == loads);
    bfly::Context uninitialized;
    assert(!uninitialized.makePlan(canonical));
    for (const auto& candidate : candidates) {
        if (candidate.state != bfly::State::Infeasible)
            assert(context.makePlan(candidate));
        if (candidate.a.localExchange != "shared")
            assert(candidate.state == bfly::State::Infeasible);
    }
    float dummy = 0;
    for (uint32_t n : {0u, 1u, 32u, 192u, 16384u}) {
        assert(plan->run(&dummy, &dummy, n, 1) == -8);
        assert(plan->runR2C(&dummy, &dummy, n, 1) == -8);
        assert(plan->runC2R(&dummy, &dummy, n, 1) == -8);
        assert(plan->prepare(n, 1) == -8);
    }
    assert(plan->run(&dummy, &dummy, 8192, 1) == -8);
    assert(plan->runC2R(&dummy, &dummy, 8192, 1) == -8);
    assert(plan->runR2C(&dummy, &dummy, 64, 1) == -8);
    for (auto run : {&bfly::Plan::run, &bfly::Plan::runR2C, &bfly::Plan::runC2R}) {
        assert((plan.get()->*run)(&dummy, &dummy, 128, 0) == -8);
        assert((plan.get()->*run)(nullptr, &dummy, 128, 1) == -8);
        assert((plan.get()->*run)(&dummy, nullptr, 128, 1) == -8);
        assert((plan.get()->*run)(&dummy, &dummy, 128, 0xffffffffu) == -8);
    }
    bfly::Metric metric;
    assert(plan->measure(1, 1, &metric) == -8);
    assert(plan->prepare(64, 0) == -8);
    assert(acl_test::allocations == 0 && acl_test::copies == 0 && acl_test::launches == 0);
    for (const auto& candidate : context.enumerate(64, 0))
        assert(candidate.state == bfly::State::Infeasible);

    // A failed allocation or copy while changing shape must invalidate the old key.
    assert(plan->prepare(64, 1) == 0);
    acl_test::failAllocation = acl_test::allocations + 2;
    assert(plan->prepare(128, 1) != 0);
    acl_test::failAllocation = 0;
    int before = acl_test::allocations;
    assert(plan->prepare(64, 1) == 0);
    assert(acl_test::allocations > before);
    acl_test::failCopy = acl_test::copies + 1;
    assert(plan->prepare(128, 1) != 0);
    acl_test::failCopy = 0;
    before = acl_test::allocations;
    assert(plan->prepare(64, 1) == 0);
    assert(acl_test::allocations > before);
    before = acl_test::allocations;
    assert(plan->prepare(64, 1) == 0);
    assert(acl_test::allocations == before);
}
