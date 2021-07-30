function test()

    net=network;
    
    net.numInputs=1;
    net.inputs{1}.size=256;
    net.numLayers=1;
    net.layers{1}.size=10;
    net.inputConnect(1)=1;
    net.biasConnect(1)=1;
    net.outputConnect(1)=1;
    net.layers{1}.transferFcn='purelin';
    net.trainFcn='trainlm';
    view(net);
   
%{
load('A.mat');
load('B.mat');
    P=A;
    T=B;
    net=train(net,P,T);
%}
T1=[1,0,0,0,0,0,0,0,0,0]';
T2=[0,1,0,0,0,0,0,0,0,0]';
T3=[0,0,1,0,0,0,0,0,0,0]';
T4=[0,0,0,1,0,0,0,0,0,0]';
T5=[0,0,0,0,1,0,0,0,0,0]';
T6=[0,0,0,0,0,1,0,0,0,0]';
T7=[0,0,0,0,0,0,1,0,0,0]';
T8=[0,0,0,0,0,0,0,1,0,0]';
T9=[0,0,0,0,0,0,0,0,1,0]';
T10=[0,0,0,0,0,0,0,0,0,1]';
T=[T1 T2 T3 T4 T5 T6 T7 T8 T9 T10];
T=[T T T T T T T T T T];
T=[T T T T T];

    load('P_final.mat');
    %load('B.mat');
    %load('P1.mat');
    %load('PerfectArial.mat');
    
    
    
    W=rand(10,256); 
    b=rand(10,1);
    net.IW{1,1}=W;
    net.b{1,1}= b;
    
    
    net.performParam.lr = 0.5; % learning rate
    net.trainParam.epochs = 1000; % maximum epochs
    net.trainParam.show = 35; % show
    net.trainParam.goal = 1e-6; % goal=objective
    net.performFcn = 'sse'; % criterion
    
    net=train(net,P,T);

    W = net.IW{1,1};
    B = net.b{1,1};
    %disp(W);
    %disp(B);
    
    a = sim(net,P);
    
    disp(a);
    
    
    
    
%{
net=perceptron;
net=configure(net,P,T);


[trainInd,valInd,testInd] = divideind(3000,1:2000,2001:2500,2501:3000);
W = net.IW{1,1};
b = net.b{1,1};
 here the test set Pt is used 
a = sim(net,Pt)
%}


return
