function [c, ceq] = omejitve(x)
n = numel(x);
c1 = -prod(x) + 0.75;
c2 = sum(x) - 7.5*n;
c = [c1; c2];
ceq = [];
end
