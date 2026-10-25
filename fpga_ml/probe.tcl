open_hw_manager
connect_hw_server -url localhost:3121
puts "TARGETS: [get_hw_targets]"
foreach t [get_hw_targets] {current_hw_target $t;open_hw_target;puts "DEVICES: [get_hw_devices]";close_hw_target}
close_hw_manager
