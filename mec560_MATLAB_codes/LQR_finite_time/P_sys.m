function dPdt = mRiccati(t, P, A, B, Q,R)

P = reshape(P , size(A)); %Convert from "n^2"-by-1 to "n"-by-"n"
dPdt = -(A.'*P + P*A - P*B*B.'*P + Q); %Determine derivative
dPdt = dPdt(:); %Convert from "n"-by-"n" to "n^2"-by-1