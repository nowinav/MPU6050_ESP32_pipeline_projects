clear;
clc;
close all;

%% =========================================================
% SERIAL CONFIGURATION
% ==========================================================

port = "/dev/ttyACM0";
baud = 115200;

s = serialport(port, baud);
configureTerminator(s, "LF");
flush(s);

fprintf("Connected to Arduino.\n");

fprintf("\n");
fprintf("============================================\n");
fprintf("         MOTOR HEALTH MONITOR\n");
fprintf("============================================\n");
fprintf("1 + ENTER  -> Start motor + analyze\n");
fprintf("0 + ENTER  -> Stop motor\n");
fprintf("q + ENTER  -> Quit\n");
fprintf("============================================\n");


%% =========================================================
% ACQUISITION PARAMETERS
% ==========================================================

N = 1024;

% IMPORTANT:
% This must match the actual Arduino sampling rate.
Fs = 500;


%% =========================================================
% COMMAND LOOP
% ==========================================================

while true

    command = input("\nCommand: ", "s");


    %% =====================================================
    % START MOTOR + ACQUIRE + ANALYZE
    % ======================================================

    if strcmp(command, "1")

        fprintf("\n");
        fprintf("============================================\n");
        fprintf("MOTOR STARTING...\n");
        fprintf("============================================\n");

        % Tell Arduino to turn motor ON
        write(s, '1', 'char');

        pause(0.5);

        fprintf("Motor ON.\n");
        fprintf("Collecting %d samples...\n", N);


        %% -------------------------------------------------
        % Clear old serial data
        % --------------------------------------------------

        flush(s);


        %% -------------------------------------------------
        % Allocate memory
        % --------------------------------------------------

        AX = zeros(N,1);
        AY = zeros(N,1);
        AZ = zeros(N,1);


        %% -------------------------------------------------
        % RECEIVE AX AY AZ
        % --------------------------------------------------

        k = 1;

        while k <= N

            line = readline(s);

            data = sscanf(char(line), '%f,%f,%f');

            if length(data) == 3

                AX(k) = data(1);
                AY(k) = data(2);
                AZ(k) = data(3);

                k = k + 1;

                if mod(k,100) == 0

                    fprintf( ...
                        "Received %d / %d samples\n", ...
                        k, N);

                end

            end

        end

        fprintf("Data acquisition complete.\n");


        %% =================================================
        % CONVERT RAW MPU6050 DATA TO g
        % ==================================================

        SENSITIVITY = 16384;

        AX = AX / SENSITIVITY;
        AY = AY / SENSITIVITY;
        AZ = AZ / SENSITIVITY;


        %% =================================================
        % REMOVE DC / GRAVITY
        % ==================================================

        AX = AX - mean(AX);
        AY = AY - mean(AY);
        AZ = AZ - mean(AZ);


        %% =================================================
        % BANDPASS FILTER
        % ==================================================

        LOW_FREQ = 5;

        HIGH_FREQ = min(200, Fs/2 - 1);

        if HIGH_FREQ > LOW_FREQ

            [b,a] = butter(4, ...
                [LOW_FREQ HIGH_FREQ] / (Fs/2), ...
                'bandpass');

            AX_f = filtfilt(b,a,AX);
            AY_f = filtfilt(b,a,AY);
            AZ_f = filtfilt(b,a,AZ);

        else

            fprintf("WARNING: Fs too low for bandpass filter.\n");

            AX_f = AX;
            AY_f = AY;
            AZ_f = AZ;

        end


        %% =================================================
        % TIME DOMAIN FEATURES
        % ==================================================

        RMS_X = rms(AX_f);
        RMS_Y = rms(AY_f);
        RMS_Z = rms(AZ_f);

        PEAK_X = max(abs(AX_f));
        PEAK_Y = max(abs(AY_f));
        PEAK_Z = max(abs(AZ_f));

        P2P_X = max(AX_f) - min(AX_f);
        P2P_Y = max(AY_f) - min(AY_f);
        P2P_Z = max(AZ_f) - min(AZ_f);

        CF_X = PEAK_X / RMS_X;
        CF_Y = PEAK_Y / RMS_Y;
        CF_Z = PEAK_Z / RMS_Z;


        %% =================================================
        % 3-AXIS VIBRATION
        % ==================================================

        VIBRATION = sqrt( ...
            AX_f.^2 + ...
            AY_f.^2 + ...
            AZ_f.^2);

        RMS_TOTAL = rms(VIBRATION);
        PEAK_TOTAL = max(VIBRATION);


        %% =================================================
        % FFT
        % ==================================================

        X = fft(AX_f);
        Y = fft(AY_f);
        Z = fft(AZ_f);

        P2_X = abs(X/N);
        P2_Y = abs(Y/N);
        P2_Z = abs(Z/N);

        P1_X = P2_X(1:N/2+1);
        P1_Y = P2_Y(1:N/2+1);
        P1_Z = P2_Z(1:N/2+1);

        P1_X(2:end-1) = 2 * P1_X(2:end-1);
        P1_Y(2:end-1) = 2 * P1_Y(2:end-1);
        P1_Z(2:end-1) = 2 * P1_Z(2:end-1);

        f = Fs * (0:N/2) / N;


        %% =================================================
        % DOMINANT FREQUENCY
        % ==================================================

        [~,idxX] = max(P1_X);
        [~,idxY] = max(P1_Y);
        [~,idxZ] = max(P1_Z);

        DOM_FREQ_X = f(idxX);
        DOM_FREQ_Y = f(idxY);
        DOM_FREQ_Z = f(idxZ);


        %% =================================================
        % DISPLAY RESULTS
        % ==================================================

        fprintf("\n");
        fprintf("============================================\n");
        fprintf("          MOTOR VIBRATION ANALYSIS\n");
        fprintf("============================================\n");

        fprintf("\nTIME DOMAIN\n");

        fprintf("AX RMS       : %.4f g\n", RMS_X);
        fprintf("AY RMS       : %.4f g\n", RMS_Y);
        fprintf("AZ RMS       : %.4f g\n", RMS_Z);

        fprintf("\nPEAK\n");

        fprintf("AX Peak      : %.4f g\n", PEAK_X);
        fprintf("AY Peak      : %.4f g\n", PEAK_Y);
        fprintf("AZ Peak      : %.4f g\n", PEAK_Z);

        fprintf("\nPEAK-TO-PEAK\n");

        fprintf("AX P2P       : %.4f g\n", P2P_X);
        fprintf("AY P2P       : %.4f g\n", P2P_Y);
        fprintf("AZ P2P       : %.4f g\n", P2P_Z);

        fprintf("\nCREST FACTOR\n");

        fprintf("AX Crest     : %.2f\n", CF_X);
        fprintf("AY Crest     : %.2f\n", CF_Y);
        fprintf("AZ Crest     : %.2f\n", CF_Z);

        fprintf("\n3-AXIS VIBRATION\n");

        fprintf("RMS          : %.4f g\n", RMS_TOTAL);
        fprintf("Peak         : %.4f g\n", PEAK_TOTAL);

        fprintf("\nDOMINANT FREQUENCY\n");

        fprintf("AX           : %.2f Hz\n", DOM_FREQ_X);
        fprintf("AY           : %.2f Hz\n", DOM_FREQ_Y);
        fprintf("AZ           : %.2f Hz\n", DOM_FREQ_Z);


        %% =================================================
        % HEALTH ASSESSMENT
        % ==================================================

        % TEMPORARY THRESHOLD
        % We will calibrate this later.

        RMS_LIMIT = 0.30;

        if RMS_TOTAL < RMS_LIMIT

            HEALTH = "NORMAL";

        elseif RMS_TOTAL < 2 * RMS_LIMIT

            HEALTH = "WARNING";

        else

            HEALTH = "HIGH VIBRATION";

        end


        fprintf("\n");
        fprintf("============================================\n");
        fprintf("          MOTOR HEALTH: %s\n", HEALTH);
        fprintf("============================================\n");


        %% =================================================
        % TIME DOMAIN PLOTS
        % ==================================================

        t = (0:N-1) / Fs;


        figure('Name','Motor Vibration - Time Domain');

        plot(t,AX_f);
        hold on;
        plot(t,AY_f);
        plot(t,AZ_f);

        xlabel("Time (s)");
        ylabel("Acceleration (g)");

        title("Motor Vibration");

        legend("AX","AY","AZ");

        grid on;


        %% =================================================
        % FFT PLOTS
        % ==================================================

        figure('Name','Motor Vibration - FFT');

        plot(f,P1_X);
        hold on;
        plot(f,P1_Y);
        plot(f,P1_Z);

        xlabel("Frequency (Hz)");
        ylabel("Magnitude");

        title("Motor Vibration FFT");

        legend("AX","AY","AZ");

        grid on;

        xlim([0 Fs/2]);


        fprintf("\nAnalysis finished.\n");
        fprintf("Motor remains ON.\n");
        fprintf("Enter 0 to stop it.\n");


    %% =====================================================
    % STOP MOTOR
    % ======================================================

    elseif strcmp(command, "0")

        write(s, '0', 'char');

        fprintf("\n");
        fprintf("============================================\n");
        fprintf("MOTOR OFF\n");
        fprintf("============================================\n");


    %% =====================================================
    % QUIT
    % ======================================================

    elseif strcmpi(command, "q")

        % Safety: turn motor OFF first
        write(s, '0', 'char');

        fprintf("\nMotor OFF.\n");
        fprintf("Closing serial connection...\n");

        break;


    else

        fprintf("Invalid command.\n");
        fprintf("Use 1 = ON + ANALYZE, 0 = OFF, q = QUIT\n");

    end

end


%% =========================================================
% CLOSE SERIAL CONNECTION
% ==========================================================

clear s;

fprintf("Serial connection closed.\n");1
