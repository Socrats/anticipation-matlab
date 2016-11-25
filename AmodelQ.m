%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    Anticipatory model.
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
% network outputs in order: prob_dict(10), prob_acc(1), Poff(1), w(1)
%******************
%clear;
%% Build network
% RRN - Elman network
rng('shuffle')
d1 = 1:2;
hidden = 100;
dnet = layrecnet(d1,hidden);
dnet.inputs{1}.size = 14;
dnet.layers{2}.size = 13;
dnet.layers{2}.transferFcn = 'logsig';
dnet = init(dnet);
dnet.trainParam.showWindow = false;
dnet.trainParam.epochs = 100;
view(dnet);

%% Online Training
% Parameters
mepoch = 10;
future = 10;
rounds = 100;
k = 0.01;
beta = 0.7;
discFactor = zeros(future,1);
% discFactor(1) = 0.5;
discFactor(1) = 1;
for t=2:future,discFactor(t)=beta.^(t-2);end;
acceptance = 1;
Pmax = 9;
Pmin = 0;
Surp = 10;
p = 0.1*ones(1,10);
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
predP = zeros(10,future);
predA = zeros(10,future);
Dactions = zeros(1,rounds);
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

for round = 1:rounds
    %% Reciever plays
    if acceptance == 0
        acceptance = 1;
    else
%         if w(:,end) == 1
%             acceptance = (a > 2);
%         elseif w(:,end) == 0.5
%             acceptance = (a > 1);
%         else
%             acceptance = (a > 0);
%         end  
        acceptance = (a > 2);
        % acceptance = (a > datasample(0:4,1,'Weights',[0.2 0.5 0.15 0.1 0.05]));
        % acceptance = (a > datasample(0:4,1));
    end
    
    %% Update reciever
    out = dnet({[Ad(:,end);Ar(:,end);Poff(:,end);w(:,end)]});
    accProb = out{1}(11);
    reciever = ind2vec(1 + 1,2);
    A = out{1}(12).*10;
    wp = out{1}(13);
       
    %% Anticipation
    % Predictions
    if acceptance
        for action=1:10
            out = dnet({[Ad(:,end);Ar(:,end);Poff(:,end);w(:,end)]});
            dictator = ind2vec(action,10);
            payoff = (10-action)./10;
            context = wp;
            X{1,1} = [dictator;reciever;payoff;context];
            predP(action,1) = payoff*10.0;  
            predA(action,1) = 1;
            for i=2:future
                out = dnet(X(1,i-1));
                probD = out{1}(1:10);
                [~, acP] = max(probD);
                dictator = ind2vec(acP,10);
                Ract = (rand() <= out{1}(11));
                recP = ind2vec(Ract + 1,2);
                acc = out{1}(11);
                if Ract
                    payoff = (10-acP)./10;
                else
                    payoff = 0;
                    dictator = zeros(10,1);
                end;
                context = out{1}(13); 
                predP(action,i) = (10-acP)*acc;  
                predA(action,i) = acc; 
                X{1,i} = [dictator;recP;payoff;context];
            end;
        end;

        utility = predP*discFactor;
        % Computing probabilities
        p = softmax(utility./k);
    end;
    %% Dictator plays
    if acceptance
        a = datasample(actions,1,'Weights',p);
        ppoff = (Surp-a)./Surp;
        action_input = a;
        Dactions(round) = a;
    else
        ppoff = 0;
        action_input = 0;
    end;
    % Compute prediction error
    errorp(round) = mse(acceptance - accProb);
    
    %% Update target micro-epoch
    % shift-left
    Adt = circshift(Adt, [0, -1]);
    Art = circshift(Art, [0, -1]);
    Pofft = circshift(Pofft, [0, -1]);
    % Update
    Adt(:,end) = p;
    Art(:,end) = acceptance;
    Pofft(:,end) = ppoff;
    index(round) = Surp*ppoff;
    
    %% Training
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
    if action_input > 0
        Ad(:,end) = ind2vec(action_input,10);
    else
        Ad(:,end) = zeros(10,1);
    end;
    Ar(:,end) = ind2vec(acceptance+1,2);
    Poff(:,end) = Pofft(:,end); 
%     if mod(round,5)==1       
%         wtt = datasample([0 0.5 1],1);        
%     end;
    w = circshift(w, [0, -1]);
    w(:,end) = 1;
    % w(:,end) = wtt 
    
    %% Statistics    
    addpoints(h1,round,errort(round));    
    addpoints(h2,round,errorp(round));
    addpoints(h3,round,index(round));
    hold on
    addpoints(h4,round,p(1));
    addpoints(h5,round,p(2));
    addpoints(h6,round,p(3));
    addpoints(h7,round,p(4));
    addpoints(h8,round,p(5));
    addpoints(h9,round,p(6));
    addpoints(h10,round,p(7));
    addpoints(h11,round,p(8));
    addpoints(h12,round,p(9));
    addpoints(h13,round,p(10));
    hold off;
    addpoints(h14,round,A);
    drawnow limitrate
    
end;
drawnow