// Legality self-test for the architecture descriptors (addendum §2 step).
// Emits machine-readable lines consumed by tests/test_descriptors.py:
//   H,<key>,<value>                      hardware profile (must match config JSON)
//   U,<core>,<min_log>,<max_log>,<ub>,<scope>,<multi_role>
//   CASE,<name>,<SUPPORTED|UNSUPPORTED>,<reason>
//   STRUCT,<name>,launches,<n>/gm_boundaries,<n>/host_assisted,<0|1>/on_chip,<0|1>
//          /abstract,<0|1>/ub_peak,<bytes>/kinds,<k1|k2|...>
// Build: g++ -std=c++17 -Iinclude tests/test_descriptors.cpp -o build/test_descriptors
#include "butterfly/descriptors.hpp"
#include <cstdio>

using namespace butterfly;

static void emit_case(const char* name, const LoweringResult& r){
  printf("CASE,%s,%s,%s\n", name, r.supported ? "SUPPORTED" : "UNSUPPORTED",
         r.reason.c_str());
}

static void emit_struct(const char* name, const LoweringResult& r){
  printf("STRUCT,%s,launches,%d,gm_boundaries,%d,host_assisted,%d,on_chip,%d,"
         "abstract,%d,ub_peak,%zu,kinds,",
         name, r.visible_launches, r.materialized_gm_boundaries,
         (int)r.host_assisted, (int)r.whole_transform_on_chip,
         (int)r.abstract_feasible, r.plan_ub_bytes);
  for(size_t i=0;i<r.launch_manifest.size();i++)
    printf("%s%s", i?"|":"", launch_kind_name(r.launch_manifest[i].kind));
  printf("\n");
}

int main(){
  const HardwareProfile& hw = builtin_hardware_profile();
  printf("H,profile_id,%s\n", hw.profile_id);
  printf("H,soc,%s\n", hw.soc);
  printf("H,aicore_num,%d\n", hw.aicore_num);
  printf("H,vector_core_num,%d\n", hw.vector_core_num);
  printf("H,l2_bytes,%zu\n", hw.l2_bytes);
  printf("H,global_mem_total_bytes,%zu\n", hw.global_mem_total_bytes);
  printf("H,ub_bytes_per_core,%zu\n", hw.ub_bytes_per_core);
  printf("H,vec_lane_fp32,%d\n", hw.vec_lane_fp32);

  const ProcessingUnitCapability unit = local_fft_capability();
  printf("U,%s,%d,%d,%zu,%s,%d\n", unit.core.c_str(), unit.min_local_log_n,
         unit.max_local_log_n, unit.ub_bytes,
         sync_scope_name(unit.synchronization_scope), (int)unit.supports_multi_role);

  // --- supported: today's G1 envelope at both ends + batch independence ---
  {
    const auto m = default_long_mapping(64, 128, hw);           // n=8192 split
    emit_case("default_8192", query_lowering({8192, 1}, m, unit, hw));
    emit_struct("default_8192", query_lowering({8192, 1}, m, unit, hw));
    emit_case("default_65536_b3", query_lowering({65536, 3}, default_long_mapping(256, 256, hw), unit, hw));
    emit_case("batch_4096_independent", query_lowering({8192, 4096}, m, unit, hw));
    emit_struct("batch_4096_independent", query_lowering({8192, 4096}, m, unit, hw));
    // DeviceGM home: the addendum §3 chain is executable and reports 6 launches.
    auto dev = m;
    dev.boundary_home = BoundaryHome::DeviceGM;
    emit_case("default_device_gm", query_lowering({8192, 1}, dev, unit, hw));
    emit_struct("default_device_gm", query_lowering({8192, 1}, dev, unit, hw));
    emit_struct("default_host_memory", query_lowering({8192, 1}, m, unit, hw));
    // Resource-abstract feasible but the runtime does not implement the tuple.
    auto cell = m;
    cell.residence = Residence::CellResident;
    emit_case("cell_resident_not_lowered", query_lowering({8192, 1}, cell, unit, hw));
    emit_struct("cell_resident_not_lowered", query_lowering({8192, 1}, cell, unit, hw));
    { auto us2 = m; us2.stage_space = 2;
      emit_case("us2_not_lowered", query_lowering(TransformSpec{8192,1}, us2, unit, hw)); }
    { auto td99 = m; td99.data_time = 99;
      emit_case("td99_not_lowered", query_lowering(TransformSpec{8192,1}, td99, unit, hw)); }
    { auto pb99 = m; pb99.pipeline_buffers = 99;
      emit_case("pipeline99_not_lowered", query_lowering(TransformSpec{8192,1}, pb99, unit, hw)); }
  }

  // --- UB resource model (PR #2 stage 3): per-kernel peaks, serial max ---
  {
    auto dev = default_long_mapping(64, 128, hw);
    dev.boundary_home = BoundaryHome::DeviceGM;
    const TransformSpec s{8192, 1};
    // ub < 98304: device chain rejected naming the transpose peak,
    // while the host chain (row FFTs only) still fits the same UB.
    { auto small = unit; small.ub_bytes = 80000;
      emit_case("ub_below_transpose_peak",
                query_lowering(s, dev, small, hw));
      emit_case("ub_host_ok_at_80000",
                query_lowering(s, default_long_mapping(64, 128, hw), small, hw)); }
    // ub == transpose peak: transpose passes, row FFT and twiddle are
    // checked next (small stages fit => supported, peak recorded).
    { auto exact = unit; exact.ub_bytes = (size_t)AB_TRANSPOSE_UB_BYTES;
      const auto r = query_lowering(s, dev, exact, hw);
      emit_case("ub_exact_transpose_peak", r);
      emit_struct("ub_exact_transpose_peak", r); }
    // ...but a 4096-point row FFT needs 191616 B > 98304 B: after the
    // transpose threshold passes, the row-FFT check rejects.
    { auto exact = unit; exact.ub_bytes = (size_t)AB_TRANSPOSE_UB_BYTES;
      auto m2 = default_long_mapping(4096, 4096, hw);
      m2.boundary_home = BoundaryHome::DeviceGM;
      emit_case("ub_rowfft_checked_after_transpose",
                query_lowering({16777216u, 1}, m2, exact, hw)); }
    // R0.1 override 穿透：mapping.row_fft_fold_d 必须进入 UB 门禁
    // （descriptor 与 launch 解析同一 (D,K)）；强制 D=1024 使行 FFT 必然溢出。
    { auto mo = default_long_mapping(64, 128, hw);
      mo.row_fft_fold_d = 1024;
      auto ub98 = unit; ub98.ub_bytes = (size_t)AB_TRANSPOSE_UB_BYTES;
      emit_case("fold_d_override_rejected",
                query_lowering(TransformSpec{8192, 1}, mo, ub98, hw)); }
  }

  // --- unsupported: explicit reasons, one per legality rule ---
  {
    auto m = default_long_mapping(64, 128, hw);
    TransformSpec s{8192, 1};

    { auto bad = s; bad.precision = Precision::FP16;
      emit_case("precision_fp16", query_lowering(bad, m, unit, hw)); }
    { auto bad = s; bad.direction = Direction::R2C;
      emit_case("direction_r2c_unmapped", query_lowering(bad, m, unit, hw)); }
    { emit_case("bad_power_of_two", query_lowering({1000, 1}, m, unit, hw)); }
    { const auto m2 = default_long_mapping(64, 8192, hw);        // stage log2 13 > 12
      emit_case("stage_outside_unit_range", query_lowering({524288, 1}, m2, unit, hw)); }
    { auto bad = m; bad.data_space = 64;
      emit_case("ud_exceeds_aiv", query_lowering(s, bad, unit, hw)); }
    { auto bad = m; bad.stage_space = 96;
      emit_case("us_exceeds_aiv", query_lowering(s, bad, unit, hw)); }
    { auto bad = m; bad.stage_partition = 3;
      emit_case("partition_mismatch", query_lowering(s, bad, unit, hw)); }
    { auto bad = m; bad.role_stages = {0, 5};
      emit_case("role_out_of_range", query_lowering(s, bad, unit, hw)); }
    { const auto m2 = default_long_mapping(256, 256, hw);       // product 65536 != 8192
      emit_case("stage_product_mismatch", query_lowering(s, m2, unit, hw)); }
    { auto bad = m; bad.boundary_layout = Layout::HalfSpectrumReal;
      emit_case("layout_real_on_c2c", query_lowering(s, bad, unit, hw)); }
    { auto small = unit; small.ub_bytes = 32768;
      const auto m2 = default_long_mapping(64, 2048, hw);        // ub_need 36864 > 32768
      emit_case("ub_overflow", query_lowering({131072, 1}, m2, small, hw)); }
    { auto bad = m; bad.residence = Residence::BlockResident;
      emit_case("block_resident_needs_multi_role", query_lowering(s, bad, unit, hw)); }
    { auto bad = m; bad.boundary_home = BoundaryHome::OnChip;
      emit_case("onchip_without_block_residence", query_lowering(s, bad, unit, hw)); }
    { const auto m2 = default_long_mapping(256, 256, hw);
      emit_case("gm_guard_overflow", query_lowering({65536, 1u<<27}, m2, unit, hw)); }
  }
  return 0;
}
