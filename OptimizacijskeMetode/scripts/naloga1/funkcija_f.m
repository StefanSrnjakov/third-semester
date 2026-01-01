function y = funkcija_f(x)
n = numel(x);
vsota = sum(cos(x).^4);
produkt = prod(cos(x).^2);
imenovalec = sqrt(sum((1:n)'.*(x.^2)));
y = abs((vsota - 2*produkt) / imenovalec);
end
