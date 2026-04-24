% Half-wave dipole at 1 GHz
d = dipole('Length', 0.15, 'Width', 0.001);
figure;
pattern(d, 1e9);               
% 3D radiation pattern
title('Half-wave Dipole @ 1 GHz');
figure;
patternElevation(d, 1e9, 0);   % 2D elevation cut
