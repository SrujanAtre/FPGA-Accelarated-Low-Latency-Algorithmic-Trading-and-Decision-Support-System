open_saif uart_activity.saif
log_saif [get_objects -r /tb_uart/dut/*]
run all
close_saif
quit
