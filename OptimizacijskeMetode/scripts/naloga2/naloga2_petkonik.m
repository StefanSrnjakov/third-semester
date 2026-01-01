st_zacetkov = 200;
lb = -10 * ones(10,1);
zg = 10 * ones(10,1);
najvecja_ploscina = -inf;
najmanjsa_ploscina = inf;
tocke_max = [];
tocke_min = [];
mozn = optimoptions('fmincon','Display','off');

for k = 1:st_zacetkov
    zacetek = lb + (zg - lb) .* rand(10,1);
    [u_max, fval_max, izhod_max] = fmincon(@(u) -ploscina_petkotnika(u), zacetek, [], [], [], [], lb, zg, @omejitve_petkotnik, mozn);
    if izhod_max > 0
        pl = -fval_max;
        if pl > najvecja_ploscina
            najvecja_ploscina = pl;
            tocke_max = u_max;
        end
    end
    [u_min, fval_min, izhod_min] = fmincon(@ploscina_petkotnika, zacetek, [], [], [], [], lb, zg, @omejitve_petkotnik, mozn);
    if izhod_min > 0
        pl2 = fval_min;
        if pl2 < najmanjsa_ploscina
            najmanjsa_ploscina = pl2;
            tocke_min = u_min;
        end
    end
end

T_max = reshape(tocke_max, 2, 5)';
T_min = reshape(tocke_min, 2, 5)';

disp('Tocke za maksimalno ploscino:')
disp(T_max)
disp('Maksimalna ploscina:')
disp(najvecja_ploscina)

disp('Tocke za minimalno ploscino:')
disp(T_min)
disp('Minimalna ploscina:')
disp(najmanjsa_ploscina)
