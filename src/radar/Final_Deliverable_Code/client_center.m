% Author: Weite Zhang (zhang.weit@northeastern.edu),
% and Jose Martinez (j.martinez-lorenzo@northeastern.edu);

% IWR1443BOOST+DCA1000EVM radar controlled through the socket msg;

% mmWaveStudio must be updated to version "02_01_00_00"

% most of the functionalities are enabled only for the non-cascaded case

% remember to set radar_initialization==1 each time you change chirp profile

clear
clc
close all
addpath('my_func')

radar_config = struct(...
    ... %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ... %%%%%%%%%% Desired Imaging Performance Parameters %%%%%%%%%%%
    ... %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    'range_max_obj', 10, ... % maximum detectable range [m]
    'range_res_obj', 0.04, ... % range resolution [m]
    'velocity_max_obj', 4, ... % absolute maximum detectable velocity [m/sec.]
    'velocity_res_obj', 0.1, ... % velocity resolution [m/sec.]
    'obj_RCS', 1, ... % RCS of the object [default = 1m^2]
    'DSP_SNR_min', 15, ... % minimum SNR required by the imaging algorithum to detect the object [default = 10dB]
    'Rx_VGA_gain', 30, ... % Rx VGA gain with all even values from 24 to 48 [default = 30dB]
    'opt_priority', [10,1,10,1], ... % optimization preority of [range_max_obj range_res_obj velocity_max_obj velocity_res_obj]: larger value indicate higher priority for optimization
    'radar_initialization', 0, ... % MM-wave front-end initialization using STUDIO APP
    ...                            % Set it to 1 whenever you modify the chirp profile
    ... %
    ... %
    ... %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ... %%%%%%%%%%%%%% Hardware and Software Settings %%%%%%%%%%%%%%%
    ... %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    'cascaded_mode', 0, ... % using 4-chip cascaded radar if 1; otherwise, set it to 0
    'enable_real_time', 1, ... % enable_real_time==1 will force num_frames==0 such that infinite number of frames will be transmitted 
    'cli_trigger_delay', 0.2, ... % interval between 'start_record' and 'stop_record' when enable_real_time==1
    'server_IP', '0.0.0.0', ... % server IP address
    'COM_NUM', 3, ... % COM port corresponding to 'XDS110 Class Application/User UART' when cascaded_mode==0
    'raw_top_dir', 'c:\radar_data\demo', ... % top directory to dave raw .bin files
    'bg_removal_server', 0, ... % apply background removal on imaging processing
    'bg_bin_file_server', '*.bin', ... % binary file of the background measurement
    'bg_removal_client', 0, ... % apply background removal on imaging processing
    'bg_bin_file_client', '*.bin', ... % binary file of the background measurement
    'server_save_multibins', 1,... % set to 1 if saving individual .bin file for each frame in the server when enable_real_time==1
    'request_raw_adc_data', 0, ... % streaming the raw ADC data to the client through socket if request_raw_adc_data==1; otehrwise, only get the imaged information
    'client_save_multibins', 1,... % set to 1 if saving individual .bin file for each frame in the client when enable_real_time==1
    'num_frames', 0, ...% number of frames within one measurement; 0 means infinite
    'frame_periodicity', 100, ... % frame periodicity in ms
    'BPM', 0, ... % using binary phase modulation if 1
    'NumMeas', 10, ... % total number of measurements: each measurement can have multiple frames; and each frame can have multiple chirps
    'window_flag', 1, ... % Hamming window for th                                                 ADC data
    'range_filtering', [0 0.2 200], ... % [whether_enable_rang_filter, range_min_to_keep, range_max_to_keep]
    'server_visualization', 1, ... % 0: no visualization in the server end; 1: visulization without saving figures; 2: visulization with saving figures
    'client_visualization', 0, ... % 0: no visualization in the client end; 1: visulization without saving figures; 2: visulization with saving figures
    'client_figure2_window', [0.05 0.2 0.1 0.1], ... % window size of the figure 2
    ... %
    ... %
    ... %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ... %%%%%%%%%% Remaining Parameters May Not Be Changed %%%%%%%%%%
    ... %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    'n_adc_samples_max', [], ... % maximum number of samples for complex 1x case per Rx chain is 2047. See Line #115.
    ... % Increase this will decrease velocity_max but increase range_res
	'sampling_rate_Sps', [], ... % maximum at 18.75 MSPS with a step size of 1 KSPS (IF maximum at 15MHz). See Line #116.
    ... % Increase this will increase velocity_max but decrease range_res 
    'idleTimeConst_min', 10e-6, ... % 10us as the minimum idle time, WHICH MAY BE SAMLLER
    'smooth_delay', 100, ... % delay time of x10ns added at the end of each chirp to ensure a smooth switching
    'post_proc_method', 2, ... 
    ... % CAS: Angle-zero-phase if 1; Angle-2D-FFT if 2
    ... % SINGLE: Angle-1D-FFT if 1; Angle-2D-FFT if 2
    'file_checkout_time', 3, ... % checking the .bin file in secs to see if the frame is available in secs 
    'delay_before_trigger', 0.8, ... % in sec, delay time after arm the DCA1000EVM and before the data capture trigger. It may also dependent on different platforms (0.2 for Desktop PC, 0.7 for miniUSB-PC)
    'InputBufferSize', 1e7, ... % socket parameter
    'OutputBufferSize', 1e4, ... % socket parameter
    'num_theta', 90, ... % elevation angle mesh size
    'num_phi', 180, ... % azimuthal angle mesh size
    'threshold_AoA', -3, ... % AoA-range: display threshold
    'zero_velocity_range', 0, ... % plot the range profile from the 2D range-Doppler map at zero-velocity
    'threshold_velocity', -30, ... % range-dopple: display threshold
    'threshold_micro_doppler', -30, ... % micro-doppler: display threshold
    'ind_tx_plot', 1, ... the index of Tx: range profile, range-doppler map
    'ind_rx_plot', 1, ... the index of Rx: range profile, range-doppler map
    'ind_loop_plot', 1, ... the index of chirp loop: range profile
    'FrequencyLimits', [-2000, 2000], ... % micro-doppler FrequencyLimits
    'range_max', [], ... % maximum detectable range [meter]
    'range_res', [], ... % range resolution [meter]
    'velocity_max', [], ... % +- maximum detectable velocity / 2  [meter/sec]
    'velocity_res', [] ... % velocity resolution [meter/sec]
    );

% check parameter correctness
if radar_config.enable_real_time == 1
    if radar_config.num_frames > 0
        warning('Current number of frames has been set to be 0 since the radar is running in real-time!')
        radar_config.num_frames = 0;
    end
elseif radar_config.enable_real_time == 0
    if radar_config.num_frames == 0
        warning('Current number of frames has been set to be 1 since the radar is not running in real-time!!')
        radar_config.num_frames = 1;
    end
end

% map user inputs to actual chirp configurations
fprintf('<strong>Desired Imaging Performance:</strong>\n');
fprintf('  Maximum detectable range        is %7.3f [m];\n', radar_config.range_max_obj);
fprintf('  Range resolution                is %7.3f [cm];\n', radar_config.range_res_obj*100);
fprintf('  Maximum detectable velocity     is \x00B1%6.3f [m/sec.];\n', radar_config.velocity_max_obj);
fprintf('  Velocity resolution             is %7.3f [cm/sec.];\n', radar_config.velocity_res_obj*100);
fprintf('  RCS of the object               is %7.3f [m^2];\n', radar_config.obj_RCS);
fprintf('  Minimum DSP SNR                 is %7.3f [dB];\n', radar_config.DSP_SNR_min);
fprintf('  Rx VGA gain                     is %7.3f [dB];\n', radar_config.Rx_VGA_gain);
fprintf('\n\n');
current_error = Inf;
ind_iteration = 1;
for ind_adc_num = 1:1:32
    for ind_adc_rate = 1:1:6
        radar_config.n_adc_samples_max = 64*ind_adc_num-1; % maximum at 64*32-1 = 2047
        radar_config.sampling_rate_Sps = 2.5e6*ind_adc_rate; % maximum at 2.5e6*6 = 15.0e6
        radar_config = chirp_para_gen(radar_config);
        
        % regularization parameter added to reflect user's preference
        loss_function = radar_config.opt_priority(1)*(radar_config.range_max_obj-radar_config.range_max)^2+...
                        radar_config.opt_priority(2)*(radar_config.range_res_obj*100-radar_config.range_res*100)^2+...
                        radar_config.opt_priority(3)*(radar_config.velocity_max_obj-radar_config.velocity_max).^2+...
                        radar_config.opt_priority(4)*(radar_config.velocity_res_obj*100-radar_config.velocity_res*100).^2;
                 
        if current_error>loss_function
            current_error = loss_function;
            % save current radar configuration structure
            iterative_radar_config = radar_config;
        end

        if ind_adc_num==1 && ind_adc_rate==1
            tt_p=tic; 
        %     count = fprintf('Adaptive optimization started with the minimum loss function = %6.3f. Elapsed time = %.3f [ms].\n\n\n', current_error, toc(tt_p)*1000);
        % else
        %     for ind_char=1:count
        %         fprintf('%c',8);
        %     end
        %     count = fprintf('Adaptive optimization finished with the minimum loss function = %6.3f. Elapsed time = %.3f [ms].\n\n\n', current_error, toc(tt_p)*1000);
        end
        ind_iteration = ind_iteration+1;
    end
end
radar_config = iterative_radar_config;
fprintf('<strong>Adaptive optimization finished with the minimum loss function = %6.3f. Elapsed time = %.3f [ms]</strong>.\n', current_error, toc(tt_p)*1000);
fprintf('\n\n');

% print and save the chirp configuration
config_printer(radar_config);

% connect to the server and run the radar measurement there
A = rstd_single_client(radar_config);