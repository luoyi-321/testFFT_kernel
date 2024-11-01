% Define the size of the FFT (must be a power of 2 for radix-2 FFTs)
N = 8;

% Initialize the input array with complex values
x = complex((0:N-1)', zeros(N, 1)); % Real part is 0 to N-1, imaginary part is 0

% Perform the FFT
Y = fft(x);

% Display the results
fprintf('FFT result:\n');
for i = 1:N
    fprintf('Y[%d] = (%.4f, %.4f)\n', i-1, real(Y(i)), imag(Y(i)));
end
