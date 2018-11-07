% Adaptation of "Sample Figure with Note" from TTU Thesis Guide.
clear all
close all

x5400 = [21, 32, 42, 117, 142]; % mm
y5400 = [81, 40, 19, 10, 4]; % mm

xaldrich = [29, 42, 58, 70, 84]; % mm
yaldrich = [80, 41, 20, 10, 4]; % mm

xscale = 5/47; % minutes/mm
yscale = 5/19; % (mg/mL)/mm

plot(5+x5400*xscale,y5400*yscale,'s-',...
    5+xaldrich*xscale,yaldrich*yscale,'d-');
xlabel('Time (minutes)')
ylabel('Concentration (mg/mL)');
legend('5400 MW','Aldrich HA');
axis([5 25 0 25]);
