`timescale 1ns / 1ps

module uart_rx #(
    parameter integer CLK_FREQ  = 125_000_000,
    parameter integer BAUD_RATE = 115200
)(
    input  wire      clk,
    input  wire      rx,

    output reg [7:0] data,
    output reg       data_valid
);

    localparam integer CLKS_PER_BIT = CLK_FREQ / BAUD_RATE;
    localparam integer HALF_BIT     = CLKS_PER_BIT / 2;

    reg rx_meta;
    reg rx_sync;

    reg [31:0] clk_count;
    reg [3:0]  bit_index;
    reg [7:0]  rx_shift;

    reg busy;

    always @(posedge clk) begin

        // Synchronizer
        rx_meta <= rx;
        rx_sync <= rx_meta;

        data_valid <= 1'b0;

        if (!busy) begin

            clk_count <= 0;
            bit_index <= 0;

            // Detect falling edge / START condition
            if (rx_sync == 1'b0) begin

                busy <= 1'b1;

                // Wait 1.5 bit periods before sampling
                // the first data bit.
                clk_count <= 0;

            end

        end

        else begin

            // First data bit is sampled after 1.5 bit periods.
            if (bit_index == 0) begin

                if (clk_count < CLKS_PER_BIT + HALF_BIT - 1) begin

                    clk_count <= clk_count + 1;

                end

                else begin

                    clk_count <= 0;

                    rx_shift[0] <= rx_sync;
                    bit_index <= 1;

                end

            end

            else begin

                // Remaining data bits every one bit period
                if (clk_count < CLKS_PER_BIT - 1) begin

                    clk_count <= clk_count + 1;

                end

                else begin

                    clk_count <= 0;

                    if (bit_index < 8) begin

                        rx_shift[bit_index] <= rx_sync;
                        bit_index <= bit_index + 1;

                    end

                    else begin

                        // Stop bit received
                        busy       <= 1'b0;
                        data       <= rx_shift;
                        data_valid <= 1'b1;
                        bit_index  <= 0;

                    end

                end

            end

        end

    end


    initial begin

        rx_meta    = 1'b1;
        rx_sync    = 1'b1;

        clk_count  = 0;
        bit_index  = 0;
        rx_shift   = 0;

        busy       = 1'b0;

        data       = 8'd0;
        data_valid = 1'b0;

    end

endmodule