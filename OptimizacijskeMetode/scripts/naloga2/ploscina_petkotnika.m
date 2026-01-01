function y = ploscina_petkotnika(u)
x = [u(1) u(3) u(5) u(7) u(9)];
y_koor = [u(2) u(4) u(6) u(8) u(10)];
y = 0.5 * abs(sum(x .* circshift(y_koor, -1) - y_koor .* circshift(x, -1)));
end
