function [S, velikost] = velikost_spozn(G)

n = size(G,1);
D = distances(G);

neveljavni = [];

for i = 1:n
    for j = i+1:n
        for v = 1:n
            if D(i,j) > D(v,i) + D(v,j)
                neveljavni = [neveljavni; i j];
                break
            end
        end
    end
end

f = -ones(n,1);

A = zeros(size(neveljavni,1), n);
b = ones(size(neveljavni,1),1);

for k = 1:size(neveljavni,1)
    i = neveljavni(k,1);
    j = neveljavni(k,2);
    A(k,i) = 1;
    A(k,j) = 1;
end

lb = zeros(n,1);
ub = ones(n,1);
intvars = 1:n;

x = intlinprog(f, intvars, A, b, [], [], lb, ub);

S = find(x > 0.5);
velikost = numel(S);

end
