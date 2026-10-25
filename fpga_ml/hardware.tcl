set root [file normalize [file dirname [info script]]]
open_hw_manager
connect_hw_server -url localhost:3121
open_hw_target
set devices [get_hw_devices -filter {PART =~ "xczu7*"}]
if {[llength $devices] != 1} {error "Expected one ZCU104 xczu7 device: $devices"}
set dev [lindex $devices 0]
current_hw_device $dev
refresh_hw_device $dev
set_property PROGRAM.FILE "$root/vivado/fpga_ml.runs/impl_1/fpga_ml_top.bit" $dev
program_hw_devices $dev
refresh_hw_device $dev
puts "ML_BITSTREAM_PROGRAMMED"
close_hw_manager
