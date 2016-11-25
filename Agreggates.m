nexec = 10;
sumProb = zeros(nexec,10);
sumFrec = zeros(nexec,10);
edges = [1 2 3 4 5 6 7 8 9 10 11];
ppof = 0;
mean_end = zeros(1,nexec);
var_end = zeros(1,nexec);

for jj=1:nexec
    % BMmodel;
    % BMmasuda;
    % Amodel
    % AmodelQ;
    Bmodel_notrain;
    % Amodel_notrain;
    sumProb(jj,:) = p;
    sumFrec(jj,:) = histcounts(Dactions,edges,'Normalization', 'probability');
    ppof = ppof + sum(index)/30;
    mean_end(jj) = mean(Dactions);
    var_end(jj) = var(Dactions);
    % sumFrec(i,:) = N./100;
end;
ppof = ppof/nexec
meanProb = sum(sumProb,1)./nexec;
meanFrec = sum(sumFrec,1)./nexec;

final_mean_end = mean(mean_end)
final_var_end = var(mean_end)
figure;
%title('Frequency of actions'),xlabel('actions'),ylabel('Frequency')
bar(1:10,meanProb)
sss = meanProb(1:5);
sss(6) = sum(meanProb(6:10));
figure;
bar(1:6,sss)
figure;
bar(1:10,meanFrec)
figure;
ssss = meanFrec(1:5);
ssss(6) = sum(meanFrec(6:10));
bar(1:6,ssss)