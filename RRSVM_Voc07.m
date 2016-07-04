classdef RRSVM_Voc07
% By: 
%     Zijun Wei (zijwei@cs.stonybrook.edu)
%     Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu), 


    methods (Static)        
        % Run voc2007, global
        function run_global(modelName, C_factor, lambda_factor)
            % Good values: C_factor: 1e4, lambda_factor: 1e-4
            classes = M_ExtremeVal.voc07classes;            
            annoDir = '/home/minhhoai/DataSets/VOCdevkit/VOC2007/ImageSets/Main/';
            if strcmpi(modelName, 'CNN_M_128') || strcmpi(modelName, 'CNN_M_1024')
                featDir = sprintf('/home/minhhoai/DataSets/Voc2007_CNN/%s', modelName);
            else
                featDir = '~/titan_work3/Voc2007/sift_bag1024';
            end
            
            tr = load(sprintf('%s/trainval.mat', annoDir), 'imIds', 'lbs', 'imSzs'); 
            tst = load(sprintf('%s/test.mat', annoDir), 'imIds', 'lbs', 'imSzs'); 
            
            startT = tic;
            fprintf('Loading train and test bags\n');
            trBags  = MIR_Voc07CNN.loadVoc2007_bag_global(tr.imIds, featDir);                                    
            tstBags = MIR_Voc07CNN.loadVoc2007_bag_global(tst.imIds, featDir);            
            fprintf('\nLoading time: %.1f seconds\n', toc(startT));
                        
            trD = cell(1, length(trBags));
            for i=1:length(trBags)
                % normalize
                trBags{i} = ML_Norm.l2norm(trBags{i}); 
                trD{i} = mean(trBags{i},2);
            end;
            
            tstD = cell(1, length(tstBags));
            for i=1:length(tstBags)
                tstBags{i} = ML_Norm.l2norm(tstBags{i}); 
                tstD{i} = mean(tstBags{i},2);
            end;
            trD  = cat(2, trD{:});
            tstD = cat(2, tstD{:});
            trD  = ML_Norm.l2norm(trD);
            tstD = ML_Norm.l2norm(tstD);                
            
            aps = zeros(length(classes),2);
            for i=1:length(classes)
                fprintf('training SVM for %s\n', classes{i});
                trLb_i  = tr.lbs(:,i);
                trIdxs_i = trLb_i ~= 0;
                trLb_i = trLb_i(trIdxs_i);
                                
                tstLb_i = tst.lbs(:,i);
                tstIdxs_i = tstLb_i ~= 0;
                tstLb_i = tstLb_i(tstIdxs_i);
                
                C = C_factor/length(trLb_i);
                svmModel = svmtrain(trLb_i, double(trD(:,trIdxs_i)'), ...
                     sprintf('-t 0 -c %g -q', C)); 
                
                w = svmModel.Label(1)*svmModel.SVs'*svmModel.sv_coef;
                b = -svmModel.rho;                
                score_i = tstD(:, tstIdxs_i)'*w + b;
                
                lambda = lambda_factor*length(trLb_i);
                [w2,b2] = ML_Ridge.ridgeReg(trD(:,trIdxs_i), trLb_i, lambda, ones(size(trLb_i)));
                score_i2 = tstD(:, tstIdxs_i)'*w2 + b2;

                aps(i,1) = ml_ap(score_i, tstLb_i, 0);  
                aps(i,2) = ml_ap(score_i2, tstLb_i, 0);  
                fprintf('%s, SVM: %.2f, LSSVM: %.2f\n', classes{i}, 100*aps(i,:));
            end
            
            fprintf('------------\n');
            for i=1:length(classes)
                fprintf('%-14s: SVM: %.2f, LSSVM: %.2f\n', classes{i}, 100*aps(i,:));
            end;
            fprintf('%-14s: SVM: %.2f, LSSVM: %.2f\n', 'Mean', 100*mean(aps));
        end;
                
        function run_lssvm_ssd(modelName, lambda_factor) 
            % lambda_factor = 5e-5
            if strcmpi(modelName, 'CNN_M_128') || strcmpi(modelName, 'CNN_M_1024')
                featDir = sprintf('~/DataSets/Voc2007_CNN/%s', modelName);
            else
                featDir = '~/titan_work3/Voc2007/sift_bag1024';
            end            
            
            addpath('~/Study/OxProjects/LSVA/src');
            classes = M_ExtremeVal.voc07classes;            
            annoDir = '~/aliases/VOCdevkit/VOC2007/ImageSets/Main/';
            
            trSet = 'trainval';
            tstSet = 'test';
            tr = load(sprintf('%s/%s.mat', annoDir, trSet), 'imIds', 'lbs', 'imSzs'); 
            tst = load(sprintf('%s/%s.mat', annoDir, tstSet), 'imIds', 'lbs', 'imSzs'); 
                        
            startT = tic;
            fprintf('Loading train and test bags\n');
            trBags  = MIR_Voc07CNN.loadVoc2007_bag_global(tr.imIds, featDir);                                    
            tstBags = MIR_Voc07CNN.loadVoc2007_bag_global(tst.imIds, featDir);            
            fprintf('\nLoading time: %.1f seconds\n', toc(startT));
            
            
            trD = cell(1, length(trBags));
            for i=1:length(trBags)
                % normalize
                trBags{i} = ML_Norm.l2norm(trBags{i}); 
                trD{i} = mean(trBags{i},2);
            end;
            trD = cat(2, trD{:});
            trD  = ML_Norm.l2norm(trD);
            
            aps = zeros(3, length(classes));
            for i=1:length(classes)
                fprintf('training SVM for %s\n', classes{i});
                trLb_i  = tr.lbs(:,i);
                trIdxs_i = find(trLb_i ~= 0);
                trLb_i = trLb_i(trIdxs_i);                
                s = ones(length(trLb_i), 1);
                
                lambda = lambda_factor*length(trLb_i);
                [w,b, ~, cvWs, cvBs] = ML_Ridge.ridgeReg_cv(trD(:,trIdxs_i), trLb_i, lambda, s);
                %[w,b] = ML_Ridge.ridgeReg(trD(:,trIdxs_i), trLb_i, lambda, s);
                
                tstLb_i = tst.lbs(:,i);
                tstIdxs_i = find(tstLb_i ~= 0);
                tstLb_i = tstLb_i(tstIdxs_i);
                                                
                % Layer 1 score
                tstL1Score = zeros(size(tstBags{1},2), length(tstIdxs_i));
                for j=1:length(tstIdxs_i)                
                    tstL1Score(:,j) = double(tstBags{tstIdxs_i(j)})'*w + b;                    
                end
                
                aps(1, i) = ml_ap(mean(tstL1Score,1)', tstLb_i, 0);            
                aps(2, i) = ml_ap(max(tstL1Score,[],1)', tstLb_i, 0);
                
                trL1Score = zeros(size(trBags{1},2), length(trIdxs_i));
                for j=1:length(trIdxs_i)                
                    trL1Score(:,j) = double(trBags{trIdxs_i(j)})'*cvWs(:,j) + cvBs(j);                    
                end
                
                trL1Score  = sort(trL1Score, 1, 'descend');
                tstL1Score = sort(tstL1Score, 1, 'descend');
                
                % Learn Layer 2                
                [w2, b2] = MLS_TrTstExp.learnLayer2(trL1Score, trLb_i, 1);
                score_2layer = tstL1Score'*w2 + b2;                                                
                                
                aps(3, i) = ml_ap(score_2layer, tstLb_i, 0);  
                fprintf('%s: mean: %.2f, max: %.2f, lssvm-ssd: %.2f\n', classes{i}, 100*aps(:, i));
            end
            
            fprintf('------------\n');
            for i=1:length(classes)
                fprintf('%-14s, mean: %.2f, max: %.2f, lssvm-ssd: %.2f\n', classes{i}, 100*aps(:,i));
            end;
            fprintf('%-14s, mean: %.2f, max: %.2f, lssvm-ssd: %.2f\n', 'mean', 100*mean(aps, 2)); 
            ml_save(sprintf('../rslt/voc2007_Nov14/ssd_%s.mat', tstSet), 'aps', aps, 'classes', classes);
        end
        
        % Model name: CNN_M_128, CNN_M_1024, sift
        % IR LSSVM
        function run_irlssvm(modelName, classIds, lambda_factor)
            trainSet = 'trainval';
            testSet = 'test';
            classes = M_ExtremeVal.voc07classes;            
            annoDir = '/home/minhhoai/DataSets/VOCdevkit/VOC2007/ImageSets/Main/';            
            if strcmpi(modelName, 'CNN_M_128') || strcmpi(modelName, 'CNN_M_1024')
                featDir = sprintf('/home/minhhoai/DataSets/Voc2007_CNN/%s', modelName);
            else
                featDir = '~/titan_work3/Voc2007/sift_bag1024';
            end
            tr = load(sprintf('%s/%s.mat', annoDir, trainSet), 'imIds', 'lbs', 'imSzs'); 
            tst = load(sprintf('%s/%s.mat', annoDir, testSet), 'imIds', 'lbs', 'imSzs'); 
            
            startT = tic;
            fprintf('Loading train and test bags\n');
            trBags  = MIR_Voc07CNN.loadVoc2007_bag_global(tr.imIds, featDir);                                    
            tstBags = MIR_Voc07CNN.loadVoc2007_bag_global(tst.imIds, featDir);            
            fprintf('\nLoading time: %.1f seconds\n', toc(startT));
                        
            trD = cell(1, length(trBags));
            for i=1:length(trBags)
                % normalize
                trBags{i} = ML_Norm.l2norm(double(trBags{i})); 
                trD{i} = mean(trBags{i},2);
            end;
            
            tstD = cell(1, length(tstBags));
            for i=1:length(tstBags)
                tstBags{i} = ML_Norm.l2norm(double(tstBags{i})); 
                tstD{i} = mean(tstBags{i},2);
            end;
            trD = cat(2, trD{:});
            tstD = cat(2, tstD{:});
            trD = ML_Norm.l2norm(trD); 
            tstD = ML_Norm.l2norm(tstD); 
            
            for kk=1:length(classIds)
                classId = classIds(kk);
                fprintf('Running for class %s\n', classes{classId});
                startT0 = tic;
                rsltFile = sprintf('../rslt/voc2007_Nov14/irlssvm6e-%s_lf-%g_%02d_%s.mat', ...
                    modelName, lambda_factor, classId, testSet);
%                 if exist(rsltFile, 'file')
%                     continue;
%                 end;

                trLb  = tr.lbs(:,classId);            
                trIdxs = trLb ~= 0;            
                trLb   = trLb(trIdxs);            

                tstLb  = tst.lbs(:,classId);
                tstIdxs = tstLb ~= 0;
                tstLb = tstLb(tstIdxs);
            
                aps = zeros(1, 2);

                % Initialization
                lambda = lambda_factor*length(trLb);
                [w,b] = ML_Ridge.ridgeReg(trD(:,trIdxs), trLb, lambda, ones(size(trLb)));

                initScore = tstD(:,tstIdxs)'*w + b;
                aps(1) = ml_ap(initScore, tstLb, 0);

                opts.initOpt = 'wb';            
                opts.w = w;
                opts.b = b;
                opts.nIter = 100;
                opts.nThread = 1;            
                                
                opts.methodName = 'IRLSSVM6';
                opts.initOpt = 'mean';
                %opts.compactConstrs = {'all'};
                [w, b, s, objVals] = MIR_IRLSSVM6.train(trBags(trIdxs), trLb, lambda, opts);                           
                tstScore = M_IRSVM.predict(tstBags(tstIdxs), w, b, s);                
                aps(2) = ml_ap(tstScore, tstLb, 0);
                                                                
                fprintf('%s, start: %5.2f, IRLSSVM6: %5.2f\n', ...
                    classes{classId}, 100*aps);
                fprintf('#non-zero s: %d\n', sum(s > 1e-3));
                
                methods = {'start', 'IRLSSVM6'};            

                elapseT = toc(startT0);                
                ml_save(rsltFile, 'aps', aps, 'methods', methods, 'objVals', objVals, 'w', w, 'b', b, 's', s, ...
                     'elapseT', elapseT);            
                fprintf('Total elapseT: %g\n', elapseT);
            end
        end;
        
        % IRLSSVM then RCS. Assume IRLSSVM have been run and saved
        function run_irlssvm_rcs(modelName, lambda_factor)
            %modelName = 'CNN_M_128';
            %lambda_factor = 5e-5;            
            trainSet = 'trainval';
            testSet = 'test';
            classes = M_ExtremeVal.voc07classes;            
            nClass = length(classes);
            annoDir = '~/aliases/VOCdevkit/VOC2007/ImageSets/Main/';            
            if strcmpi(modelName, 'CNN_M_128') || strcmpi(modelName, 'CNN_M_1024')
                featDir = sprintf('~/DataSets/Voc2007_CNN/%s', modelName);
            else
                featDir = '~/titan_work3/Voc2007/sift_bag1024';
            end            
            tr = load(sprintf('%s/%s.mat', annoDir, trainSet), 'imIds', 'lbs', 'imSzs'); 
            tst = load(sprintf('%s/%s.mat', annoDir, testSet), 'imIds', 'lbs', 'imSzs'); 
            
            startT = tic;
            fprintf('Loading train and test bags\n');
            trBags  = MIR_Voc07CNN.loadVoc2007_bag_global(tr.imIds, featDir);                                    
            tstBags = MIR_Voc07CNN.loadVoc2007_bag_global(tst.imIds, featDir);            
            fprintf('\nLoading time: %.1f seconds\n', toc(startT));
            
            [trScores, tstScores, aps] = deal(cell(1, nClass));
            for kk=1:nClass
                fprintf('Computing trScores and tstScores for %s\n', classes{kk});
%                 rsltFile = sprintf('../rslt/voc2007_Oct14/irlssvm-weightS3_%s_lambda-%g_%02d_%s.mat', ...                    
%                     modelName, lambda, kk, testSet);
                
                rsltFile = sprintf('../rslt/voc2007_Nov14/irlssvm6c-%s_lf-%g_%02d_%s.mat', ...
                        modelName, lambda_factor, kk, testSet);
                    
                [w, b, s, aps_i] = ml_load(rsltFile, 'w', 'b', 's', 'aps'); 
                aps{kk} = aps_i;
                tstScores{kk} = M_IRSVM.predict(tstBags, w, b, s);
                trScores{kk} = M_IRSVM.predict(trBags, w, b, s);
            end;
            aps = cat(1, aps{:});
            trScores = cat(2, trScores{:})';
            tstScores = cat(2, tstScores{:})';
            
            for kk=1:nClass
                trLb  = tr.lbs(:,kk);            
                trIdxs = trLb ~= 0;            
                trLb   = trLb(trIdxs);            

                tstLb  = tst.lbs(:,kk);
                tstIdxs = tstLb ~= 0;
                tstLb = tstLb(tstIdxs);
                
                rcsTrD = trScores(:, trIdxs);
                rcsTstD = tstScores(:, tstIdxs);
                
%                 rcsTrD = [rcsTrD(kk,:); sort(rcsTrD([1:kk-1,kk+1:nClass],:), 1, 'descend')];
%                 rcsTstD = [rcsTstD(kk,:); sort(rcsTstD([1:kk-1,kk+1:nClass],:), 1, 'descend')];
                rcsTrD = [rcsTrD(kk,:); sort(rcsTrD([1:kk-1,kk+1:nClass],:), 1, 'descend'); ...
                    rcsTrD([1:kk-1,kk+1:nClass],:)];
                rcsTstD = [rcsTstD(kk,:); sort(rcsTstD([1:kk-1,kk+1:nClass],:), 1, 'descend'); ...
                    rcsTstD([1:kk-1,kk+1:nClass],:)];


                svmModel = svmtrain(trLb, rcsTrD', sprintf('-t 0 -c 0.1 -q')); 
%                 svmModel = svmtrain(trLb, rcsTrD', sprintf('-t 0 -c %g -w1 %g -w-1 %g -q', ...
%                     100, 1/sum(trLb ==1), 1/sum(trLb == -1)));
                w3 = svmModel.Label(1)*svmModel.SVs'*svmModel.sv_coef;
                b3 = -svmModel.Label(1)*svmModel.rho;
                
                rcsScore = rcsTstD'*w3 + b3;
                aps(kk,3) = ml_ap(rcsScore, tstLb, 0);                
                fprintf('%-11s, start: %.2f, IRLSSVM6: %.2f, RCS: %.2f\n', classes{kk}, 100*aps(kk,:));
            end;
            
            fprintf('%-11s, start: %.2f, IRLSSVM6: %.2f, RCS: %.2f\n', 'Mean', 100*mean(aps)); 
            
            keyboard;
            
        end
        
        % MISVM 
        function run_MISVM(modelName, classIds, C_factor)
            startT0 = tic;                                    
            trainSet = 'trainval';
            testSet  = 'test';
            if strcmpi(modelName, 'CNN_M_128') || strcmpi(modelName, 'CNN_M_1024')
                featDir = sprintf('~/DataSets/Voc2007_CNN/%s', modelName);
            else
                featDir = '~/titan_work3/Voc2007/sift_bag1024';
            end            

            classes = M_ExtremeVal.voc07classes;            
            annoDir = '~/aliases/VOCdevkit/VOC2007/ImageSets/Main/';                        
            tr = load(sprintf('%s/%s.mat', annoDir, trainSet), 'imIds', 'lbs', 'imSzs'); 
            tst = load(sprintf('%s/%s.mat', annoDir, testSet), 'imIds', 'lbs', 'imSzs'); 
            
            startT = tic;
            fprintf('Loading train and test bags\n');
            trBags  = MIR_Voc07CNN.loadVoc2007_bag_global(tr.imIds, featDir);                                    
            tstBags = MIR_Voc07CNN.loadVoc2007_bag_global(tst.imIds, featDir);            
            fprintf('\nLoading time: %.1f seconds\n', toc(startT));
            
            trD = cell(1, length(trBags));
            for i=1:length(trBags)
                % normalize
                trBags{i} = ML_Norm.l2norm(double(trBags{i})); 
                trD{i} = mean(trBags{i},2);
            end;
            
            tstD = cell(1, length(tstBags));
            for i=1:length(tstBags)
                tstBags{i} = ML_Norm.l2norm(double(tstBags{i})); 
                tstD{i} = mean(tstBags{i},2);
            end;
            trD = cat(2, trD{:});
            tstD = cat(2, tstD{:});
            trD = ML_Norm.l2norm(trD);
            tstD = ML_Norm.l2norm(tstD);
            
            for kk=1:length(classIds)
                classId = classIds(kk);
                rsltFile = sprintf('../rslt/voc2007_Nov14/MISVM_%s_cf-%g_%02d_%s.mat', modelName, C_factor, classId, testSet);
                fprintf('Running for class %s\n', classes{classId});

                trLb   = tr.lbs(:,classId);            
                trIdxs = trLb ~= 0;            
                trLb   = trLb(trIdxs);            
    
                tstLb  = tst.lbs(:,classId);
                tstIdxs = tstLb ~= 0;
                tstLb = tstLb(tstIdxs);
                        
                aps = zeros(1, 2);

                C = C_factor/length(trLb);
                % Initialization                
                svmModel = svmtrain(trLb, double(trD(:, trIdxs)'), sprintf('-t 0 -c %g -q', C));
                w = svmModel.Label(1)*svmModel.SVs'*svmModel.sv_coef;
                b = -svmModel.rho;

                initScore = tstD(:, tstIdxs)'*w + b;
                aps(1) = ml_ap(initScore, tstLb, 0);

                opts.initOpt = 'wb';            
                opts.w = w;
                opts.b = b;
                opts.nIter1 = 10;
                opts.nIter2 = 10;           
                opts.nThread = 1;         
                opts.dispFunc = [];

                opts.isMISVM = 1;
                opts.methodName = 'MISVM';
                [w, b, s, objVals, iterInfo] = M_IRSVM.train(trBags(trIdxs), trLb, C, opts);           
                
                tstScore2 = M_IRSVM.predict(tstBags(tstIdxs), w, b, s); 
                aps(2) = ml_ap(tstScore2, tstLb, 0);               

                fprintf('%s, start: %5.2f, MISVM: %5.2f\n', ...
                    classes{classId}, 100*aps);
                methods = {'start', 'MISVM'};            

                elapseT = toc(startT0);
                ml_save(rsltFile, 'aps', aps, 'methods', methods, 'w', w, 'b', b, 's', s, ...
                    'elapseT', elapseT, 'objVals', objVals, 'iterInfo', iterInfo);            
            end
        end;

        % Run voc2007, gt instance classification
        function run_gt(modelName, lambda_factor)
            % lambda_factor = 5e-5
            if strcmpi(modelName, 'CNN_M_128') || strcmpi(modelName, 'CNN_M_1024')
                featDir = sprintf('~/DataSets/Voc2007_CNN/%s', modelName);
            else
                featDir = '~/titan_work3/Voc2007/sift_bag1024';
            end            
            
            addpath('~/Study/OxProjects/LSVA/src');
            classes = M_ExtremeVal.voc07classes;            
            annoDir = '~/aliases/VOCdevkit/VOC2007/ImageSets/Main/';
            
            trSet = 'trainval';
            tstSet = 'test';
            tr = load(sprintf('%s/%s.mat', annoDir, trSet), 'imIds', 'lbs', 'imSzs'); 
            tst = load(sprintf('%s/%s.mat', annoDir, tstSet), 'imIds', 'lbs', 'imSzs'); 
            
            aps = zeros(1, length(classes));
            for i=1:length(classes)
                fprintf('training Gt SVM for %s\n', classes{i});
                [trD, trLb] = MIR_Voc07.load_gt_global(tr.imIds, tr.lbs(:,i), i, featDir);
                [tstD, tstLb] = MIR_Voc07.load_gt_global(tst.imIds, tst.lbs(:,i), i, featDir);
                trD = ML_Norm.l2norm(trD);
                tstD = ML_Norm.l2norm(tstD);
                
                lambda = lambda_factor*length(trLb);
                [w, b] = ML_Ridge.ridgeReg(trD, trLb, lambda, ones(size(trLb)));                
                tstScore = tstD'*w + b;
                aps(i) = ml_ap(tstScore, tstLb, 0);
                fprintf('%-12s: %.2f\n', classes{i}, 100*aps(i));
            end
            
            fprintf('------------\n');
            for i=1:length(classes)
                fprintf('%-14s: InstanceAP: %.2f\n', classes{i}, 100*aps(i));
            end;
            fprintf('%-14s: InstanceAP: %.2f\n', 'Mean', 100*mean(aps));
            
        end

              
        function dispRslt_irlssvm(modelName, lambda_factor)
            aps = zeros(2, 20);
            %modelName = 'CNN_M_1024';
            testSet = 'test';
            classes = M_ExtremeVal.voc07classes;            
            for classId = 1:20
                try
                    rsltFile = sprintf('../rslt/voc2007_Nov14/irlssvm6c-%s_lf-%g_%02d_%s.mat', ...
                        modelName, lambda_factor, classId, testSet);
                    [aps_i, objVals, s] = ml_load(rsltFile, 'aps', 'objVals', 's');
                    aps(:,classId) = aps_i;
                    fprintf('%-12s, init: %5.2f, IRLSSVM6: %5.2f, objVal: %6.2f, #non-zero s: %2d\n', ...
                        classes{classId}, 100*aps(:, classId), objVals(end), sum(s > 1e-3));
                catch
                    fprintf('%-12s\n', classes{classId});
                end
            end;
            fprintf('%-12s, init: %5.2f, IRLSSVM6: %5.2f\n', 'mean', 100*mean(aps,2));
            
            for classId = 1:20
                try
                    rsltFile = sprintf('../rslt/voc2007_Nov14/irlssvm6c-%s_lf-%g_%02d_%s.mat', ...
                        modelName, lambda_factor, classId, testSet);
                    [elapseT, objVals] = ml_load(rsltFile, 'elapseT', 'objVals');                    
                    fprintf('%-12s, nIter: %f, elapseT: %f\n', classes{classId}, length(objVals)/2, elapseT);       
                    %s = ml_load(rsltFile, 's');                    
                    %fprintf('%-12s, s = ', classes{classId}); fprintf('%g ', s(s > 1e-3)); fprintf('\n');                    
                catch
                    fprintf('%-12s\n', classes{classId});
                end
            end;            
        end;
        
        function dispRslt_misvm(modelName, C_factor)
            aps = zeros(2, 20);
            %modelName = 'CNN_M_1024';
            testSet = 'test';
            classes = M_ExtremeVal.voc07classes;            
            for classId = 1:20
                try
                    rsltFile = sprintf('../rslt/voc2007_Nov14/misvm_%s_cf-%g_%02d_%s.mat', ...
                        modelName, C_factor, classId, testSet);
                    [aps_i, objVals, s] = ml_load(rsltFile, 'aps', 'objVals', 's');
                    aps(:,classId) = aps_i;
                    fprintf('%-12s, init: %5.2f, MISVM: %5.2f, objVal: %6.2f, #non-zero s: %2d\n', ...
                        classes{classId}, 100*aps(:, classId), objVals(end), sum(s > 1e-3));
                catch
                    fprintf('%-12s\n', classes{classId});
                end
            end;
            fprintf('%-12s, init: %5.2f, MISVM: %5.2f\n', 'mean', 100*mean(aps,2));
        end;        
        
        % Load the global vector
        % featOpt: global or mean
        function Ds = load_global(imIds, featDir, featOpt)
            Ds = cell(1, length(imIds));
            for i=1:length(imIds);
                ml_progressBar(i, length(imIds));            
                if strcmpi(featOpt, 'global')
                    A = load(sprintf('%s/%06d.mat', featDir, imIds(i)), 'globalFeat');
                    Ds{i} = A.globalFeat;
                elseif strcmpi(featOpt, 'mean')
                    A = load(sprintf('%s/%06d.mat', featDir, imIds(i)), 'objRectFeats');
                    Ds{i} = mean(A.objRectFeats,2);
                else
                    error('unknown option');
                end;
            end;               
        end;
        
        % load instances
        % imLb: vector of 1 or -1, to sample one positive or one negative instance per bag
        % clsId: class Id
        function [D, lb] = load_gt_global(imIds, imLb, clsId, featDir) 
            posImIds = imIds(imLb == 1);
            negImIds = imIds(imLb == -1);
            
            posDs = cell(1, length(posImIds));
            for i=1:length(posImIds)
                ml_progressBar(i, length(posImIds));
                A = load(sprintf('%s/%06d.mat', featDir, posImIds(i)), ...
                    'gtRectFeats', 'gtRects', 'globalFeat');
                idxs = find(A.gtRects(5,:) == clsId); 
                posDs{i} = cat(1, A.globalFeat, A.gtRectFeats(:, randsample(idxs, 1)));
            end;
            
            negDs = cell(1, length(negImIds));
            nNegPerIm = 1;
            for i=1:length(negImIds)
                ml_progressBar(i, length(negImIds));
                
                A = load(sprintf('%s/%06d.mat', featDir, negImIds(i)), 'objRectFeats', 'globalFeat');                
                idxs = randsample(size(A.objRectFeats,2), nNegPerIm); % sample 10 random one only
                negDs{i} = cat(1, A.globalFeat, A.objRectFeats(:, idxs));
            end
            posDs = cat(2, posDs{:});            
            negDs = cat(2, negDs{:});
            
            D = cat(2, posDs, negDs);
            lb = [ones(size(posDs,2), 1); -ones(size(negDs,2),1)];
        end;
        
        
        % Calculate intersection over union for a set of objRects and each gtRect in turn
        % gtRects: 4*n matrix, 
        % iou: scalar for the intersection over union of all objRects and all gtRects
        % ious: 1*n values for intersection over union for all objRects and each gtRects
        function [iou, ious] = calIOU(objRects, gtRects)
            imH = max([objRects(4,:), gtRects(4,:)]);
            imW = max([objRects(3,:), gtRects(3,:)]);
            im = zeros(imH, imW);
            for i=1:size(objRects,2)
                im(objRects(2,i):objRects(4,i), objRects(1,i):objRects(3,i)) = 1;
            end;
            
            gtIms = zeros(imH, imW, size(gtRects,2));
            for i=1:size(gtRects,2)
                gtIms(gtRects(2,i):gtRects(4,i), gtRects(1,i):gtRects(3,i), i) = 1;
            end;
            im = im(:);
            gtIms = reshape(gtIms, imH*imW, size(gtRects,2));
            gtIm = sum(gtIms, 2);
            iou = sum(and(im, gtIm))/sum(or(im, gtIm));
            
            im = repmat(im, 1, size(gtRects,2));
            intersection = sum(and(im, gtIms), 1);
            union = sum(or(im, gtIms), 1);
            ious = intersection./union; 
            
        end;

    end    
end

