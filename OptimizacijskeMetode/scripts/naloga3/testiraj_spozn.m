load grafi3
fprintf("Test za C7x7:\n");
[S1, v1] = velikost_spozn(C7x7);
disp(S1); disp(v1);

fprintf("Test za C8x8:\n");
[S2, v2] = velikost_spozn(C8x8);
disp(S2); disp(v2);

fprintf("Test za G:\n");
[S3, v3] = velikost_spozn(G);
disp(S3); disp(v3);
