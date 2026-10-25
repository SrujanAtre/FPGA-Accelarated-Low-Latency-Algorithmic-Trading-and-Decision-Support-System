set root [file normalize [file dirname [info script]]]
cd $root
create_project fpga_ml "$root/vivado" -part xczu7ev-ffvc1156-2-e -force
foreach f [glob "$root/rtl/*.v"] {
 if {![string match "tb_*" [file tail $f]]} {add_files [list $f]}
}
set_property include_dirs [list "$root/rtl"] [get_filesets sources_1]
add_files [glob "$root/rtl/*.mem"]
set_property top fpga_ml_top [get_filesets sources_1]
add_files -fileset constrs_1 [list "$root/rtl/fpga_ml_top.xdc"]
set_property STEPS.SYNTH_DESIGN.ARGS.FLATTEN_HIERARCHY rebuilt [get_runs synth_1]
launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {error "Synthesis failed"}
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {error "Implementation failed"}
open_run impl_1
report_utilization -file "$root/reports/utilization.rpt"
report_timing_summary -file "$root/reports/timing.rpt"
report_power -file "$root/reports/power.rpt"
report_drc -file "$root/reports/drc.rpt"
write_checkpoint -force "$root/reports/ml_routed.dcp"
