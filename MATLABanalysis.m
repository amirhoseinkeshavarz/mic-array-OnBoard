clc
close all
clear all

for i = 0:4
    name = ['channel', num2str(i), '.mat'];
    load(name, 'mydata')
    channels(i+1, :) = mydata;
end

mics_locs = [(-2:2).'*0.061, zeros(5, 1)];

theta = (0:179).' * pi/180;
vMat = [cos(theta), sin(theta)];
% figure; plot(steering(:, 1), steering(:, 2), 'ko'); axis equal
% hold on
% plot(mics_locs(:, 1), mics_locs(:, 2), 'gs')
%
fs = 40e3;
c = 343;
for i = 1:5
    for j = 1:5
        steering(i, j, :) = -round((fs/c)*dot(repmat(mics_locs(i, :) - mics_locs(j, :), size(vMat, 1), 1) , vMat, 2));
    end
end

steeringReshaped = reshape(steering,size(steering,1) * size(steering,2),[]);
[~,signalIdx] = find(abs(channels(1,:))>1);

channels = channels(:,signalIdx(1)-30:signalIdx(1) + 100);
channelsFft = fft(channels,[],2);
matchedFft = conj(channelsFft(1:end,:)).*channelsFft(1,:);
matched = fftshift(ifft(matchedFft,[],2),2);
[~,maxIdx] = max(db(matched),[],2);
maxRelative = maxIdx - maxIdx' ;
maxRelativeReshaped = reshape(maxRelative,size(maxRelative,1)*size(maxRelative,2) , []);
pattern = sum(abs(maxRelativeReshaped - steeringReshaped));
% figure; plot(abs(pattern))
[~,phiEstimated] = min(abs(pattern))

diff(maxIdx);
% figure; plot(db(matched.'))
%
figure; plot(channels')

% % figure; plot(mydata(:, 1:5))
%
% sound(mydata, 40e3)

% diff([cursor_info.DataIndex])
