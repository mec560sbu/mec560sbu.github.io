% Determine optimal K
function optK = LQR_k(A, B, Q, R)

P = care(A,B,Q,R);
K = inv(R)*B'*P';
optK = K;