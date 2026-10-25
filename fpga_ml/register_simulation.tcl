set root [file normalize [file dirname [info script]]]
open_project "$root/vivado/fpga_ml.xpr"
set existing [get_files -quiet -of_objects [get_filesets sources_1] *tb_ml_core.v]
if {[llength $existing]} {move_files -fileset sim_1 $existing}
foreach name {tb_uart.v tb_workloads.v} {
 if {![llength [get_files -quiet */$name]]} {add_files -fileset sim_1 [list "$root/rtl/$name"]}
}
set_property include_dirs [list "$root/rtl"] [get_filesets sim_1]
set_property top tb_ml_core [get_filesets sim_1]
update_compile_order -fileset sim_1
close_project
