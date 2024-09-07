clear; clc;

mdl = 'Flexible_Fishtail_CTLSys_RL';
agentBlk = [mdl '/Subsystem/RL Agent'];

numObs = 5;
obsInfo = rlNumericSpec([numObs 1]);
obsInfo.Name = 'observations';
obsInfo.Description = 'angle, deflection_S, deflection_M, deflection_L, target';
numObservations = obsInfo.Dimension(1);

numAct = 1;
actInfo = rlNumericSpec([numAct 1],'LowerLimit',-1,'UpperLimit',1);
actInfo.Name = 'action';
actInfo.Description = 'voltage';
numActions = actInfo.Dimension(1);

env = rlSimulinkEnv(mdl,agentBlk,obsInfo,actInfo);

Ts = 20e-3;
Tf = 12;
maxepisodes = 200;
maxsteps = ceil(Tf/Ts);
StopTrainingValue = 0;
SaveAgentValue = -50;

rng(0)

current_episode = 0;
episode = current_episode + maxepisodes;

%% *******************    DDPG    *********************
critic_learning_rate = 0.005;
actor_learning_rate = 0.0005;

criticLayerSizes = [32 16];
obsPath = [featureInputLayer(prod(obsInfo.Dimension),Name="obsInLyr")
            fullyConnectedLayer(criticLayerSizes(1))
            reluLayer
            fullyConnectedLayer(criticLayerSizes(2),Name="obsOutLyr")];
actPath = [featureInputLayer(prod(actInfo.Dimension),Name="actInLyr")
            fullyConnectedLayer(criticLayerSizes(2),Name="actOutLyr")];
commonPath = [additionLayer(2,'Name','add')
                reluLayer
                fullyConnectedLayer(1)];
criticNet = layerGraph(obsPath);
criticNet = addLayers(criticNet, actPath);
criticNet = addLayers(criticNet, commonPath);
criticNet = connectLayers(criticNet,"obsOutLyr","add/in1");
criticNet = connectLayers(criticNet,"actOutLyr","add/in2");
% figure(1);
% plot(criticNet);
criticNet = dlnetwork(criticNet);
critic = rlQValueFunction(criticNet,obsInfo,actInfo,...
    ObservationInputNames="obsInLyr", ...
    ActionInputNames="actInLyr");

actorLayerSizes = [12 6];
actorNet = [featureInputLayer(prod(obsInfo.Dimension))
            fullyConnectedLayer(actorLayerSizes(1))
            reluLayer
            fullyConnectedLayer(actorLayerSizes(2))
            reluLayer
            fullyConnectedLayer(prod(actInfo.Dimension))];
actorNet = dlnetwork(actorNet);
% figure(2);
% plot(actorNet);
actor = rlContinuousDeterministicActor(actorNet,obsInfo,actInfo);

agentOptions = rlDDPGAgentOptions;
agentOptions.SampleTime = Ts;
agentOptions.DiscountFactor = 0.99;
agentOptions.MiniBatchSize = 64;
agentOptions.ExperienceBufferLength = 1e3;
agentOptions.TargetSmoothFactor = 1e-3;
agentOptions.SaveExperienceBufferWithAgent = false ;
agentOptions.CriticOptimizerOptions.LearnRate=critic_learning_rate;
agentOptions.CriticOptimizerOptions.GradientThreshold=1;
agentOptions.ActorOptimizerOptions.LearnRate=actor_learning_rate;
agentOptions.ActorOptimizerOptions.GradientThreshold=1;

agent = rlDDPGAgent(actor,critic,agentOptions);

%% *****************    Training progress    ******************
trainOpts = rlTrainingOptions(...
     'MaxEpisodes',maxepisodes, ...
     'MaxStepsPerEpisode',maxsteps, ...
     'ScoreAveragingWindowLength',10, ...
     'Verbose',true, ...
     'Plots','training-progress',...
     'StopTrainingCriteria','AverageReward',...
     'StopTrainingValue',StopTrainingValue,...
     'SaveAgentCriteria','EpisodeReward',...
     'SaveAgentValue',SaveAgentValue,...
     'StopOnError','on',...
     'SaveAgentDirectory','Agent');

trainOpts.UseParallel = true;                    
trainOpts.ParallelizationOptions.Mode = 'async';
trainingoptions.Parallelizationoptions.WorkerRandomSeeds = -1;
trainOpts.ParallelizationOptions.StepsUntilDataIsSent = 32;
trainOpts.ParallelizationOptions.DataToSendFromWorkers = 'Experiences';

trainingStats = train(agent,env,trainOpts);
agent_filename = ['Agent/Agent_' num2str(episode) '.mat'];
save(agent_filename, 'agent');

