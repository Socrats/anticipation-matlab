%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Reactive model.
%    Copyright (C) {year}  {name of author}
%
%    This program is free software: you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.
%
%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with this program.  If not, see <http://www.gnu.org/licenses/>.
%   
%    Author: Elias Fernández - eliferna@vub.ac.be
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
% network inputs in order: dicator(10), reciever(2), Poff(1), w(1)
% network outputs in order: prob_dict(9), prob_acc(1), Poff(1), w(1)
%******************
%clear;
%% Build network
% RRN - Elman network with BM model
rng('shuffle')
d1 = 1:1;
hidden = 100;
dnet = layrecnet(d1,hidden);
dnet.inputs{1}.size = 14;
dnet.layers{2}.size = 13;
dnet.layers{2}.transferFcn = 'logsig';
dnet = init(dnet);
dnet.trainParam.showWindow = false;
dnet.trainParam.epochs = 100;
% view(dnet);

%% Parameters
mepoch = 10;
rounds = 500;
Pmax = 9;
Pmin = 0;
Surp = 10;
A = 9;
l = 1;
p = 0.1*ones(1,10);
a = 6;
payoff = Surp-a;
action_input = a;
acceptance = 1;
actions = 1:10;
nactions = 10;
% Inputs
Ad = zeros(10,mepoch);
Ar = zeros(2,mepoch);
Poff = zeros(1,mepoch);
w = ones(1,mepoch);
% Targets
wt = w;
Art = ones(1,mepoch);
Adt = zeros(10,mepoch);
Pofft = zeros(1,mepoch);
% Variables
window = zeros(1,mepoch);
index = zeros(1,rounds);
errort = zeros(1,rounds);
errorp = zeros(1,rounds);
Dactions = zeros(1,rounds);
pavg = zeros(10,rounds);
% t0: Initialization
reciever = ind2vec((rand() < 0.5) + 1,2);
Poff(:,end) = 0.5;
a = 5;
Ad(:,end) = ind2vec(floor((1-Poff(:,end))*10),10);
Ar(:,end) = reciever;
% Define plots
ax1 = subplot(2,1,1);axis([0 rounds 0 1])
h1 = animatedline;
ax2 = subplot(2,1,2);axis([0 rounds 0 1])
h2 = animatedline;
title(ax1,'Training performance'),xlabel(ax1,'epochs'),ylabel(ax1,'mse')
title(ax2,'Prediction performance'),xlabel(ax2,'epochs'),ylabel(ax2,'mse')
figure;
h3 = animatedline;axis([0 rounds 0 10])
title('Dictator payoff'),xlabel('epochs'),ylabel('Payoff')
figure;
title('Actions probability'),xlabel('epochs'),ylabel('Probability')
h4 = animatedline('Color','r');axis([0 rounds 0 1])
h5 = animatedline('Color','y');
h6 = animatedline('Color','m');
h7 = animatedline('Color','c');
h8 = animatedline('Color','g');
h9 = animatedline('Color','b');
h10 = animatedline('Color','k');
h11 = animatedline('Color','b','LineStyle',':');
h12 = animatedline('Color','b','LineStyle','--');
h13 = animatedline('Color','k','LineStyle','-.');
legend('a=1','a=2','a=3','a=4','a=5','a=6','a=7','a=8','a=9','a=10',...
    'Location','northoutside','Orientation','horizontal');
figure;
h14 = animatedline;axis([0 rounds 0 10])
title('Aspiration level'),xlabel('epochs'),ylabel('Aspiration')

%% Algorithm
for round=1:rounds   
    %% Reciever plays
    if acceptance == 0
        acceptance = 1;
    else
        %if w(:,end) == 1
        %    acceptance = (a > 2);
        %elseif w(:,end) == 0.5
        %    acceptance = (a > 1);
        %else
        %    acceptance = (a > 0);
        %end  
        acceptance = (a > 2);
        % acceptance = (a > datasample(0:4,1,'Weights',[0.2 0.5 0.15 0.1 0.05]));
        % acceptance = (a > datasample(0:4,1));
    end
    
    %% Select next action
    if action_input > 0
        dictator = ind2vec(action_input,10);
    else
        dictator = zeros(10,1);
    end;
    reciever = ind2vec(acceptance+1,2);
    context = w(:,end);
    sss = payoff./10;
    inpsim = {[dictator;reciever;sss;context]};
    out = dnet(inpsim);
    payoffP = out{1}(12).*10;
    probD = out{1}(1:10);
    probD = probD./sum(probD);
    probR = out{1}(11);
    
    if acceptance
        a = datasample(actions,1,'Weights',probD);
        payoff = Surp-a;
        Dactions(round) = a;
        action_input = a;
    else
        payoff = 0;
        Dactions(round) = 0;
        action_input = 0;
    end;

    % Compute prediction error
    errorp(round) = mse(acceptance - probR);
    
    %% Calculate probabilities
    A = payoffP;
    s = (payoff - A)/max(abs((Pmax - A)),abs((Pmin-A)));
    nsel = actions(actions~=a);
    if s >= 0 
        p(a) = p(a) + (1-p(a)).*s.*l;
    else
        p(a) = p(a) + p(a).*s.*l;
    end;
    if sum(p(nsel)) > 0
        p(nsel) = (p(nsel)*(1-p(a)))./sum(p(nsel));
    else
        p(nsel) = (1-p(a))./(nactions-1);
    end;
    
    pavg(:,round) = p;
    
    %% Update target micro-epoch
    % shift-left
    Adt = circshift(Adt, [0, -1]);
    Art = circshift(Art, [0, -1]);
    Pofft = circshift(Pofft, [0, -1]);
    % Update
    Adt(:,end) = p;
    Art(:,end) = acceptance;
    Pofft(:,end) = (payoff/10 + (round-1)*Pofft(:,end-1))/round;
    index(round) = payoff;
    
    %% Training
    %if round < 20
        window = max(1,mepoch-round+1):mepoch;
        inputs = [Ad(:,window);Ar(:,window);Poff(:,window);w(:,window)];
        targets = [Adt(:,window);Art(:,window);Pofft(:,window);w(:,window)];
        for i=1:400
            [dnet,Ys,Es,Xf,Yf,tr]  = adapt(dnet,mat2cell(inputs, size(inputs,1), ...
                ones(1,size(inputs,2))),mat2cell(targets, size(targets,1), ...
                ones(1,size(targets,2))));
        end;
        errort(round) = mse(Es);
    
    %% Update Input micro-epoch
    % shift-left
    Ad = circshift(Ad, [0, -1]);
    Ar = circshift(Ar, [0, -1]);
    Poff = circshift(Poff, [0, -1]);
    % Update
    if action_input > 0;
        Ad(:,end) = ind2vec(action_input,10);
    else
        Ad(:,end) = zeros(10,1);
    end;
    Ar(:,end) = ind2vec(acceptance+1,2);
    Poff(:,end) = Pofft(:,end); 
    % if mod(round,5)==1       
    %     wtt = datasample([0 0.5 1],1);        
    % end;
    w = circshift(w, [0, -1]);
    w(:,end) = 1;
    % w(:,end) = wtt
    
    %% Statistics
    addpoints(h1,round,errort(round));
    addpoints(h2,round,errorp(round));
    addpoints(h3,round,index(round));
    hold on
    addpoints(h4,round,probD(1));
    addpoints(h5,round,probD(2));
    addpoints(h6,round,probD(3));
    addpoints(h7,round,probD(4));
    addpoints(h8,round,probD(5));
    addpoints(h9,round,probD(6));
    addpoints(h10,round,probD(7));
    addpoints(h11,round,probD(8));
    addpoints(h12,round,probD(9));
    addpoints(h13,round,probD(10));
    hold off;
    addpoints(h14,round,A);
    drawnow limitrate
end;
drawnow