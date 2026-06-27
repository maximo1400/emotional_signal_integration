#set page(paper: "a4")
#set text(font: "New Computer Modern", size: 11pt)
#set heading(numbering: "1.")

= DREAMER DB Adaptation: Signal Processing Pipeline

The conversion from raw EEG voltage ($mu "V"$) to frequency band power rigorously mimics the Emotiv EPOC+ signal processing chain. Here is the mathematical breakdown of each frame's processing:

== Slew-Rate Clipping
To prevent high-frequency artifacts introduced by sudden extreme voltage spikes, the sample-to-sample difference is calculated and clamped to a maximum $delta$ (30 $mu "V"$).

$ Delta x[n] = x[n] - x[n-1] $
$ Delta x_"clipped"[n] = max(-delta, min(delta, Delta x[n])) $

The signal is then reconstructed via cumulative sum:
$ x_"clipped"[n] = sum_(i=0)^n Delta x_"clipped"[i] $

== High-Pass Filter (0.5 Hz)
A 2nd-order Butterworth IIR high-pass filter removes DC drift and ultra-low frequency sweat artifacts.

$ y[n] = sum_(k=0)^2 b_k x_"clipped"[n-k] - sum_(k=1)^2 a_k y[n-k] $

== Epoching
The continuous signal is divided into overlapping frames:
- *Window size* ($N$): 256 samples (2.0 seconds at 128 Hz)
- *Hop size*: 16 samples (0.125 seconds)

== Robust DC Removal (Interquartile Mean)
To center the signal around zero without being skewed by large localized artifacts within an epoch, the Interquartile Mean (IQM) is subtracted instead of a standard arithmetic mean.

$ Q_1, Q_3 = text("25th and 75th percentiles of the epoch ") X $
$ X_"IQR" = { x in X | Q_1 <= x <= Q_3 } $
$ "IQM" = 1 / |X_"IQR"| sum_(x in X_"IQR") x $
$ X_"centered" = X - "IQM" $

== Hann Windowing
A raised cosine window is applied to minimize spectral leakage at the edges of the finite epoch. The Emotiv implementation uses a scaling factor of 2.0.

$ w[n] = 0.5 ( 1 - cos((2 pi (n+1)) / (N+1)) ) $
$ X_"windowed"[n] = X_"centered"[n] times 2.0 times w[n] $

== Power Spectrum (FFT)
The Fast Fourier Transform (FFT) is used to transition to the frequency domain.

$ X(k) = sum_(n=0)^(N-1) X_"windowed"[n] e^(-i 2 pi k n / N) $

The absolute power for each frequency bin $k$ is calculated by squaring the magnitude and dividing by the window length $N$:

$ P(k) = |X(k)|^2 / N $

== Band Aggregation and Logarithmic Scaling
The powers of the frequency bins that fall within each designated frequency band (e.g., Theta: 4--8 Hz) are summed together.

$ P_"band" = sum_(f_k in [f_"low", f_"high")) P(k) $

To output in logarithmic scale (closer to human perception / standard feature representation), a base-10 logarithm is applied with a tiny epsilon to avoid $log(0)$:

$ P_"out" = log_10 (P_"band" + 10^(-20)) $
