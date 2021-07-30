function assoc_mem_purelin()
    
  
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
    
    %P 256x500 Inputs
    load('P_final.mat');
    %Perfect 256x10 Targets
    load('PerfectArial.mat');
    %P_test 256x50 Test Inputs
    load('P_test.mat');
    %P_2 256x250 Inputs
    load('P_2.mat');
    %T 10x750 Targets
    load('T_2.mat');
    
    load('P_FINAL_3.mat');
   
    T = [T T(:,1:250)];
    
    
    %256x750 targets for filter
    assoc_T = [Perfect Perfect Perfect Perfect Perfect ]
    assoc_T = [assoc_T assoc_T assoc_T assoc_T assoc_T assoc_T assoc_T assoc_T assoc_T assoc_T assoc_T assoc_T assoc_T assoc_T assoc_T]
    assoc_T = [assoc_T assoc_T(:,1:250)];
    
    
    W_pseudo = assoc_T*pinv(P_FINAL_3);
    
    assoc_output = W_pseudo*P_FINAL_3;

    
    
    
    net=network;
    
    net.numInputs=1;
    net.inputs{1}.size=256;
    net.numLayers=1;
    net.layers{1}.size=10;
    net.inputConnect(1)=1;
    net.biasConnect(1)=1;
    net.outputConnect(1)=1;
    
    %disp(option);
    %view(net);
    
          net.layers{1}.transferFcn='purelin';
            net.trainFcn='trainscg';
            net.inputWeights{1}.learnFcn ='trainscg';
            net.biases{1}.learnFcn = 'trainscg';
            net.divideFcn= 'dividerand'; % divide the data randomly 
            net.divideparam.trainRatio = 85/100;    % Training ratio
            net.divideparam.valRatio = 15/100;  % Validation ratio
            net.divideparam.testRatio = 0; % Testing Ratio
            W=rand(10,256); 
            b=rand(10,1);
            net.IW{1,1}=W;
            net.b{1,1}= b;
            net.performParam.lr = 0.5; % learning rate
            net.trainParam.epochs = 1000; % maximum epochs
            net.trainParam.show = 35; % show
            net.trainParam.goal = 1e-6; % goal=objective
            net.performFcn = 'sse'; % criterion
            %net = configure(net, assoc_output, T);
            [net tr]=train(net,assoc_output,T);
            
            val_assoc_T = [Perfect Perfect Perfect Perfect Perfect ];

            val_W_pseudo = val_assoc_T*pinv(P_test);

            val_assoc_output = val_W_pseudo * P_test;


            a = sim(net,val_assoc_output);

            result = a;

            for i=1:size(a,2)
                for j=1:size(a,1)
                    result(j,i)=abs(a(j,i))/max(abs(a(:,i)));
                end
            end

            %disp(a);

            sim_target=[T1 T2 T3 T4 T5 T6 T7 T8 T9 T10];
            sim_target=[sim_target sim_target sim_target sim_target sim_target];

            figure;
            plotconfusion(sim_target,result);
            figure;
            plotperform(tr);
   
   return