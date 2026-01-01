n_vrednosti = [3 5 10];
st_zacetkov = 200;

for k = 1:numel(n_vrednosti)
    n = n_vrednosti(k);
    sp_meja = zeros(n,1);
    zg_meja = 10*ones(n,1);
    najboljsa_vrednost = -inf;
    najboljsa_tocka = [];
    for j = 1:st_zacetkov
        zacetek = sp_meja + (zg_meja - sp_meja).*rand(n,1);
        fun = @(x) -funkcija_f(x);
        [x_opt, fval, izhod] = fmincon(fun, zacetek, [], [], [], [], sp_meja, zg_meja, @omejitve);
        if izhod > 0
            vrednost = -fval;
            if vrednost > najboljsa_vrednost
                najboljsa_vrednost = vrednost;
                najboljsa_tocka = x_opt;
            end
        end
    end
    fprintf('n = %d\n', n);
    disp('najboljsa tocka x*:');
    disp(najboljsa_tocka');
    fprintf('vrednost f(x*): %.6f\n\n', najboljsa_vrednost);
end
