
import numpy as np
import pandas as pd
from scipy.linalg import inv
from scipy.linalg import det
from scipy.linalg import cholesky
from math import pi
import os
import time

class NWFlowEstimator:
    def train(self, data_path, target_name, feature_name, feature_type, feature_weight=[], start=0, length=-1, model_path='NW_model.txt', max_iter=100, rglr=1e-1, learning_rate=0.005):
        feature_name = feature_name.split(':')
        feature_type = feature_type.split(':')
        feature_weight = feature_weight.split(':')
        num_of_feature = len(feature_name)
        cols = feature_name[:]
        cols.append(target_name)
        df = pd.read_csv(data_path)
        data_len = len(df)
        if length == -1:
           length = data_len
        
        if feature_weight == []:
           feature_weight = [1.0/num_of_feature] * num_of_feature
        
        data = np.array(df.loc[start:start+length-1,cols].astype(float))
        
        idx_sr, target1, target_mean1, D_nm, D_mm, theta_gp1, eta_gp1, Theta_gp1, Eta_gp1, LOO1 = self.nw_training_cplt_2(data, -1, feature_type, feature_weight, max_iter, rglr, learning_rate)
        idx_sr_data, target2, target_mean2, D_nm, D_mm, theta_gp2, eta_gp2, Theta_gp2, Eta_gp2, LOO2 = self.nw_training_cplt_2(data, -1, feature_type, feature_weight, max_iter, rglr, learning_rate, idx_sr)
        
        theta1 = np.exp(theta_gp1)
        delta1 = []
        for i in range(0,num_of_feature):
            delta1.append(np.exp(eta_gp1[i]))
        K_nm1 = NWOperation.kernel_gauss_nw_cplt(D_nm, theta1, delta1)
        K_mm1 = NWOperation.kernel_gauss_nw_cplt(D_mm, theta1, delta1)
        
        Lambda1 = np.real(inv(K_mm1 + rglr*np.identity(len(idx_sr),dtype=float)))
        alpha1 = np.dot(np.transpose(K_nm1), target1)
        alpha1 = np.dot(Lambda1, alpha1)
        
        yi = np.ones((length,1), dtype = float)
        beta1 = np.dot(np.transpose(K_nm1), yi)
        beta1 = np.dot(Lambda1, beta1)
        
        
        
        theta2 = np.exp(theta_gp2)
        delta2 = []
        for i in range(0, num_of_feature):
            delta2.append(np.exp(eta_gp2[i]))
            
        K_nm2 = NWOperation.kernel_gauss_nw_cplt(D_nm, theta2, delta2)
        K_mm2 = NWOperation.kernel_gauss_nw_cplt(D_mm, theta2, delta2)
        
        Lambda2 = np.real(inv(K_mm2 + rglr*np.identity(len(idx_sr), dtype=float)))
        alpha2 = np.dot(np.transpose(K_nm2), target2)
        alpha2 = np.dot(Lambda2, alpha2) 

        beta2 = np.dot(np.transpose(K_nm2), yi)
        beta2 = np.dot(Lambda2, beta2)
        
        
        model = NWModel()
        model.setTargetName(target_name)
        model.setFeatureName(feature_name)
        model.setFeatureType(feature_type)
        model.setFeatureWeight(feature_weight)
        model.setMaxIter(max_iter)
        model.setRglr(rglr)
        model.setLearningRate(learning_rate)
        model.setIdxSr(idx_sr_data)
        model.setTargetMean1(target_mean1)
        model.setTargetMean2(target_mean2)
        model.setLambda1(Lambda1)
        model.setLambda2(Lambda2)
        model.setAlpha1(alpha1)
        model.setAlpha2(alpha2)
        model.setBeta1(beta1)
        model.setBeta2(beta2)
        model.setTheta1(theta_gp1[0][0])
        model.setTheta2(theta_gp2[0][0])
        model.setEta1(eta_gp1)
        model.setEta2(eta_gp2)
        model.setSampleSize(length)
        model.setThetaGp1(Theta_gp1)
        model.setThetaGp2(Theta_gp2)
        model.setEtaGp1(Eta_gp1)
        model.setEtaGp2(Eta_gp2)
        model.setSimTable1([])
        model.setSimTable2([])
        model.save(model_path)
        return model
    def nw_training_cplt_2(self, input, target_pos, feature_type, feature_weight, max_iter, rglr, lda_learning, idx_sr_pre=[]):
        input = np.array(input)
        y = input[:, target_pos]
        Len = len(y)
        y = np.reshape(y,(Len,1))
        
        if idx_sr_pre !=[]:
           y = np.log(y)
        
        target = y
        target_mean = np.mean(y)
        
        pre_theta = 0
        theta = np.log(1)
        
        pre_eta = []
        eta = []
        num_of_feature = len(feature_type)
        
        for i in range(0, num_of_feature):
            pre_eta.append(0)
            eta.append(np.log(0.5))
        
        Eta = np.zeros((num_of_feature, max_iter), dtype = float)
        Eta[:,0] = np.array(eta).reshape(num_of_feature, 1)[:,0]
        Theta = [theta]

        feature_list =  range(num_of_feature + 1)
        if target_pos < 0:
           target_pos = len(feature_list) + target_pos
        
        feature_list = feature_list[0:target_pos] + feature_list[target_pos + 1:]  ###
        idx_sr = idx_sr_pre[:]
        if idx_sr == []:
           # Len_sr = np.round(Len/3)
           # idx = np.random.permutation(Len)
           # idx_sr = idx[0: Len_sr]
           Len_sr = np.round(Len/3)
           idx = np.argsort(input[:, target_pos])
           idx_sr = idx[::3]
           
        data = input[:, feature_list]
        data_sr =  data[idx_sr, :]
        
        D_nm = NWOperation.nw_k_cplt(data, data_sr, feature_type)
        
        D_mm = NWOperation.nw_k_cplt(data_sr, data_sr, feature_type)
        
        LOO = [self.nw_leaveoneout_cplt2(target, D_nm, D_mm, theta, eta, rglr)]
        
        curr_iter = 0
        stepsize = 0.001
        while curr_iter < max_iter:
              
            pre_eta = eta[:]
            pre_theta = theta
              
            Old_LOO = self.nw_leaveoneout_cplt2(target, D_nm, D_mm, pre_theta, pre_eta, rglr)
              
            for i in range(0, num_of_feature):
                pre_eta_stepsize = pre_eta[:]
                pre_eta_stepsize[i] =  pre_eta_stepsize[i] + stepsize
                eta[i] =  pre_eta[i] - lda_learning*(self.nw_leaveoneout_cplt2(target, D_nm, D_mm, pre_theta, pre_eta_stepsize, rglr) - Old_LOO)/stepsize

            Eta[:,curr_iter] = np.array(eta).reshape(num_of_feature,1)[:,0]
            
            theta = pre_theta - lda_learning*(self.nw_leaveoneout_cplt2(target, D_nm, D_mm, pre_theta+stepsize, pre_eta, rglr) - Old_LOO)/stepsize
            Theta.append(theta)
             
            LOO.append(Old_LOO)
             
            curr_iter = curr_iter + 1
             
            error1 = np.abs(theta - pre_theta)
            error2 = []
             
            for i in range(0, len(eta)):
                error2.append(np.abs(eta[i] - pre_eta[i]))
            error2 = max(error2)
             
            if error1 < 0.001 and error2 < 0.001:
               break
             
        if idx_sr_pre != []:
            idx_sr = data_sr
             
        format_eta = []
        for a in range(len(eta)):
            format_eta.append(float(eta[a][0]))
        format_Theta = []
        for a in Theta:
            format_Theta.append(float(a))
        return idx_sr, target, target_mean, D_nm, D_mm, theta, format_eta, format_Theta, Eta, LOO
             
    def update(self, data_path, model_path, start =  0, length = -1, overwrite = True): 
         old_model = NWModel()
         old.model.load(model_path)
         df = pd.read_csv(data_path)
         data_len = len(df)
         if length == -1: 
             length = data_len
         cols = old.model.getFeatureName()[:]
         cols.append(old_model.getTargetName())
         update_date = np.array(df.loc[start:start+length-1, cols].astype(float))
         
         theta1 = np.exp(old_model.getTheta1())
         eta1 = np.exp(old_model.getEta1())
         Lambda11 = old_model.getLambda1()
         alpha1 = old_model.getAlpha1()
         beta1 = old_model.getBeta1()
         
         for k in range(0,length):
             y_new = update_data[k,-1].astype(float)
             y_new = np.array(y_new)
             y_new = np.reshape(y_new,(1,1))
             sample_data = [update_data[k, 0:-1]]
             D_star = NWOperation.nw_k_cplt(sample_data, old_model.getIdxSr(), old_model.getFeatureType())
             K_star = NWOperation.kernel_gauss_nw_cplt(D_star, theta1, eta1)
             
             K = K_star.transpose()
             alpha1, beta1 = NWOperation.add_new_info(K, y_new, Lambda1, alpha1, beta1)
             
         theta2 = np.exp(old_model.getTheta2())
         eta2  = np.exp(old_model.getEta2())
         Lambda2 = old_model.getLambda2()
         alpha2 = old_model.getAlpha2()
         beta2 = old_model.getBeta2()
         
         for k in range(0,length):
             y_new = update_data[k,-1].astype(float)
             y_new = y_new ** 0.5
             y_new = np.array(y_new)
             y_new = np.reshape(y_new, (1,1))
             sample_data = [update_data[k,0:-1]]
             D_star =  NWOperation.nw_k_cplt(sample_data, old_model.getIdxSr(), old_model.getFeatureType())
             K_star = NWOperation.kernel_gauss_nw_cplt(D_star, theta2, eta2)
             K = K_star.transpose()

             alpha2, beta2 = NWOperation.add_new_info(K, y_new, Lambda2, alpha2, beta2)
             
         old_model.setAlpha1(alpha1)
         old_model.setAlpha2(alpha2)
         old_model.setLambda1(Lambda1)
         old_model.setLambda2(Lambda2)
         
         
         update_model_path  = model_path
         
         if overwrite == False:
            update_model_path += '.updated'
         old_model.save(update_model_path)
         
         return old_model
        
    def cross_validate(self, data_path, target_name, feature_name, feature_type, feature_weight, start = 0, length= -1,training_set_perc = 0.5, max_iter = 100, rglr = 1e-1, learning_rate = 0.005, model_path = 'cross_validate_model.txt'):
        raw_data = pd.read_csv(data_path)
        file_len = len(raw_data)
        
        if length == -1: 
           length = file_len
          
        train_len = int(length * train_set_perc)
        model = self. train(data_path, target_name, feature_name, feature_type, feature_weight, start, train_len, model_path, max_iter, rglr, learning_rate)
        model.save(model_path)
        
        f_mean=model.predict(data_path, train_len, (length - train_len), 'cross_validate_result.txt')
        
        thrsh_true = np.mean(raw_data.loc[start: train_len, target_name].astype(float))
        test_target = np.array(raw_data.loc[train_len: length-1, target_name].astype(float)).reshape(length-train_len, 1)

        real_label = np.zeros(length-train_len, dtype=np.integer)
        lable_test = real_label[:] 
        TP = 0
        FP = 0
        mean_elephant = 0 
        mean_mice = 0
        
        print test_target.shape
        for i in range(0, length -  train_len):
            if test_target[i,0] >= thrsh_true:
                 real_label[i] = 1
                 mean_elephant = mean_elephant + test_target[i,0]
            else: 
                 mean_mice = mean_mice + test_target[i,0]
          
            if f_mean[i] >= thrsh_true: 
                 label_test[i] = 1 

            if real_label[i] == 1 and label_test[i] == 1:
                 TP = TP + 1
             
            if  real_lable[i] == 0 and label_test[i] == 1: 
                 FP = FP + 1
        TPR = float(TP)/float(sum(real_label))
        FNR = 1 - TPR 
        
        FPR = float(FP)/float(len(test_target)-sum(real_label))
        TNR = 1 - FPR

        print real_label
        print label_test
        
        print 'The TPR is', TPR 
        print 'The FNR is', FNR
        print 'The FPR is', FPR
        print 'The TNR is', TNR
        
    
  
    def nw_k_cplt2(self, data, data_sr, feature_type):
        
        len1 = len(data)
        len2 = len(data_sr)
        xx1 = np. transpose(data)
        xx2 = np.transpose(data_sr)

        temp = []
        for x in xx1: 
             temp.append(x.tolist()) 
        xx1 =  temp

        temp=[]
        for x in xx2:
            temp.append(x.tolist()) 
        xx2 = temp
 
        num_of_feature = len(feature_type)
        K = []
        for i in range(0, num_of_feature):
            K_k = np.zeros((len1,Len2), dtype = float)
            K.append(K_k)
        dist_x1_x2 = 0.0

        for i in range(0, len1):
            for j in range(0, len2):
                for k in range(0, num_of_feature):
                     Type = feature_type[k]
                     x1 = xx1[k]
                     x2 = xx2[k]
                     if Type == 'numeric':
                         dist_x1_x2 = (x1[i]-x2[j])**2/np.abs(x1[i]*x2[j])
                     elif Type == 'IP':
                         dist_x1_x2 = (self.dist_IP(x1[i],x2[j]))**2
                     elif Type == 'Port':
                         dist_x1_x2 = (self.dist_port(x1[i], x2[j]))**2
                     elif Type == 'Categorical':
                          dist_x1_x2 = (self.dist_protocol(x1[i],x2[j]))**2
                     else:
                          dist_x1_x2 = 0.0
                     K[k][i][j] = dist_x1_x2 
        return K
   
    def nw_leaveoneout_cplt2(self, target, D_nm, D_mm, logtheta, logdelta, rglr):
         y = target
         Len = len(y)
         
         yi = np.ones((Len,1), dtype = float)
         
         Len_rr = len(D_mm[0])
         
         num_of_feature = len(logdelta)
         
         theta = np.exp(logtheta)
         delta=[]
         for i in range(0, num_of_feature):
             delta.append(np.exp(logdelta[i]))
             
         K_nm = NWOperation.kernel_gauss_nw_cplt(D_nm, theta, delta)
         K_mm = NWOperation.kernel_gauss_nw_cplt(D_mm, theta, delta)
         
         I_rr = np.identity(Len_rr, dtype = np.float)
        
         inv_K = np.real(inv(K_mm + rglr * I_rr))
         
         Q = K_nm
         
         BLK = np.dot(inv_K, np.transpose(Q))
         
         Est = np.zeros((Len,1), dtype = float)
         
         template_y = np.ones((Len,1), dtype = float)
         template_yi = np.ones((Len,1), dtype = float)
         
         for k in range(0, Len):
             template_y[:] = y[:]
             template_yi[:] = yi[:]
             template_y[k,0] = 0
             template_yi[k,0] = 0
             
             template_Q = Q[k,:]
             
             est = np.dot(template_Q, np.dot(BLK, template_y)) / np.dot(template_Q, np.dot(BLK, template_yi))
             
             Est[k, 0] = est
             
         LOO = np.log(np.dot(np.transpose(y-Est), y-Est)/np.var(y))
             
         return LOO

    def kernel_gauss_nw_cplt2(self, dist, theta, delta):
        Len = len(dist)
        m,n = dist[0].shape
       
        y = np.ones((m,n), dtype = float)

        for i in range(0,Len):
            y = y * np.exp(-dist[i]/2/delta[i])
        y = theta * y
        
        return y

class NWModel: 
     
    def predict(self, predict_file, start = 0, length = -1, result_file = 'NWresult.txt'):
         
        tic = time.time()
        df = pd.read_csv(predict_file)
        data_len = len(df)
        if length == -1:
           length = data_len
        
        data = np.array(df.loc[start:start+length-1,self.getFeatureName()].astype(float))
        D_star = NWOperation.nw_k_cplt(data,self.getIdxSr(),self.getFeatureType())
        
        theta = np.exp(self.getTheta2())
        
        num_of_feature = len(self.getFeatureName())
        
        eta = np.exp(self.getEta2())
        K_star = NWOperation.kernel_gauss_nw_cplt(D_star, theta, eta)

        f_mean = np.dot(K_star, self.getAlpha2())/np.dot(K_star, self.getBeta2())
        f_mean = f_mean ** 2

        # Idx_redo = []
        # for i in range(len(f_mean)):
             # if f_mean[i][0] > self.getTargetMean1():
                # Idx_redo.append(i)
        # if Idx_redo !=[]:
            # data_retry = data[Idx_redo, :]
            # D_star =  NWOperation.nw_k_cplt(data_retry, self.getIdxSr(), self.getFeatureType())
            # theta = np.exp(self.getTheta1())
            # eta = np.exp(self.getEta1())
            # K_star = NWOperation.kernel_gauss_nw_cplt(D_star, theta, eta)
            
            # f_mean_redo = np.dot(K_star, self.getAlpha1())
            # f_mean_redo = self.getTargetStd1() * f_mean + self.getTargetMean1()
        # pos = 0

        # for i in Idx_redo: 
            # f_mean[i][0] =  max(f_mean_redo[pos][0], f_mean[i][0])
            # pos += 1/rglr*I
        
        fw = open(result_file, 'wb')
        f_mean = (i[0] for i in f_mean)
        for i in f_mean:
            fw.write(str(i).strip()+'\n')
        fw.close()
        toc = time.time()
        print "The time for prediction is", toc-tic
        return f_mean
        
        
    def predictSingle(self, sample, sample_id, result_file = 'NWsingleResult.txt'):
         #tic = time.time()
         test_data = np.array([[float(i) for i in sample.split(':')]])
         
         D_star = NWOperation.nw_k_cplt(test_data, self.getIdxSr(), self.getFeatureType()) 
         
         theta = np.exp(self.getTheta2()) 
         
         num_of_feature = len(self.getFeatureName())
         eta = np.exp(self.getEta2())
         K_star = NWOperation.kernel_gauss_nw_cplt(D_star, theta, eta)
         
         f_mean = np.dot(K_star, self.getAlpha2())/ np.dot(K_star, self.getBeta2())
         f_mean = np.exp(f_mean)
         
         # if f_mean[0][0] > self.getTargetMean1():
            # D_star =  NWOperation.nw_k_cplt(test_data, self.getIdxSr(), self.getFeatureType())
            # theta = np.exp(self.getTheta1())
            # eta = np.exp(self.getEta1())
            
            # K_star = NWOperation.kernel_gauss_nw_cplt(D_star, theta, eta)

            # f_mean_redo = np.dot(K_star, self.getAlpha1())
            # f_mean_redo = self.getTargetStd1() * f_mean_redo + self.getTargetMean1()

            # f_mean[0][0] = max(f_mean_redo[0][0], f_mean[0][0])

         # if result_file !='':
            # if os.path.exists(result_file):
               # fw = open(result_file)
               # if len(fw.readlines()) >= 10000:
                  # fw.close()
                  # new_result = result_file + time.strftime('.%m%d_%H_%M_%S',time.localtime(time.time())) + '.txt'
                  # os.rename(result_file, new_result)
               # else:
                  # fw.close()
            # fw = open(result_file,'ab')
            # for i in f_mean:
                # fw.write(str(sample_id) + '\t' + sample + '\t' + str(i[0]).strip() + '\n')
            # fw.close()
         # toc = time.time()
         # print "The time for prediction is", toc - tic
         return f_mean[0][0]
    
    def predictSingle_2(self, sample, sample_id, result_file = 'NWsingleResult.txt'):
         #tic = time.time()
         test_data = np.array([[int(i) for i in sample.split(':')]])
         
         #simTalbe1 = self.getSimTable1()
         #simTable2 = self.getSimTable2()
         
         Len_sr = len(self.getSimTable1()[0][0][:])
         
         K_star_1= np.zeros((1,Len_sr), dtype = float)
         K_star_2 = np.zeros((1,Len_sr), dtype = float)
         num_of_feature = len(self.getFeatureName())
         
         
        
         for k in range(0, num_of_feature):
             K_star_1 = K_star_1 + (self.getSimTable1()[k][test_data[0,k]][:])/num_of_feature
             K_star_2 = K_star_2 + (self.getSimTable2()[k][test_data[0,k]][:])/num_of_feature
         
         K_star_1 = np.exp(self.getTheta1()) * (K_star_1 ** num_of_feature)
         K_star_2 = np.exp(self.getTheta2()) * (K_star_2 ** num_of_feature)
         
         f_mean_1 = np.dot(K_star_1, self.getAlpha1())/np.dot(K_star_1, self.getBeta1())
       
         
         f_mean_2 = np.dot(K_star_2, self.getAlpha2())/np.dot(K_star_2, self.getBeta2())
         f_mean_2 = np.exp(f_mean_2)
         
         
         if f_mean_2 >= np.exp(self.getTargetMean2()) or f_mean_1 <=0:
           f_mean = max(f_mean_1, f_mean_2)
         else:
           f_mean = min(f_mean_1, f_mean_2)

         # toc = time.time()            
         # if result_file !='':
            # if os.path.exists(result_file):
               # fw = open(result_file)
               # if len(fw.readlines()) >= 10000:
                  # fw.close()
                  # new_result = result_file + time.strftime('.%m%d_%H_%M_%S',time.localtime(time.time())) + '.txt'
                  # os.rename(result_file, new_result)
               # else:
                  # fw.close()
            # fw = open(result_file,'ab')
            # for i in f_mean:
                # fw.write(str(sample_id) + '\t' + sample + '\t' + str(i[0]).strip() + '\n')
            # fw.close()
        
         # print "The time for prediction is", toc - tic
         return f_mean
    
    def update_online(self, sample, y_new):
        
        update_data = np.array([[int(i) for i in sample.split(':')]])
        num_of_feature = len(self.getFeatureName())
        
        Len_sr = len(self.getSimTable1()[0][0][:])
        
        K_star_1 = np.zeros((1,Len_sr), dtype = float)
        K_star_2 = np.zeros((1,Len_sr), dtype = float)
        
        for k in range(0, num_of_feature):
            K_star_1 = K_star_1 + (self.getSimTable1()[k][update_data[0,k]][:])/num_of_feature
            K_star_2 = K_star_2 + (self.getSimTable2()[k][update_data[0,k]][:])/num_of_feature
            
        K_star_1 = np.exp(self.getTheta1()) * (K_star_1 ** num_of_feature) 
        K_star_2 = np.exp(self.getTheta2()) * (K_star_2 ** num_of_feature)
        
        K_star_1 = K_star_1.transpose()
        alpha1, beta1, target_mean1, str = NWOperation.add_new_info(K_star_1, y_new, self.getLambda1(), self.getAlpha1(), self.getBeta1(), self.getTargetMean1(), self.getSampleSize()) 
        
        self.setAlpha1(alpha1)
        self.setBeta1(beta1)
        self.setTargetMean1(target_mean1)

        K_star_2 = K_star_2.transpose()

        alpha2, beta2, target_mean2, str = NWOperation.add_new_info(K_star_2, np.log(y_new), self.getLambda2(), self.getAlpha2(), self.getBeta2(), self.getTargetMean2(), self.getSampleSize()) 
        
        self.setAlpha2(alpha2)
        self.setBeta2(beta2)
        self.setTargetMean2(target_mean2)
        
        self.setSampleSize(str)

        
    def load(self, model_path):
        f = open(model_path)
        f.readline()
        self.setTargetName(f.readline().strip()) 
        
        f.readline()
        self.setFeatureName(f.readline().strip().split(','))

        f.readline()
        self.setFeatureType(f.readline().strip().split(','))
 
        f.readline();
        parts = f.readline().strip().split(',');
        self.setFeatureWeight([float(i) for i in parts])

        f.readline()
        self.setMaxIter(int(f.readline().strip())) 

        f.readline();
        self.setRglr(float(f.readline().strip()))

        f.readline()
        self.setLearningRate(float(f.readline().strip()))
        
        Idx_sr_size = f.readline().strip().split(':')[-1].split(',')
        m = int(Idx_sr_size[0])
        n = int(Idx_sr_size[1])
        Idx_sr_array = np.array(f.readline().strip().split(','),dtype=np.float)
        self.setIdxSr(Idx_sr_array.reshape(m,n))
        
        lambda1_size = f.readline().strip().split(':')[-1].split(',')
        m = int(lambda1_size[0])
        n = int(lambda1_size[1])
        lambda1_array = np.array(f.readline().strip().split(','),dtype = np.float)
        self.setLambda1(lambda1_array.reshape(m,n))

        lambda2_size = f.readline().strip().split(':')[-1].split(',')
        m = int(lambda2_size[0])
        n = int(lambda2_size[1])
        lambda2_array = np.array(f.readline().strip().split(','),dtype = np.float)
        self.setLambda2(lambda2_array.reshape(m,n))
                
        alpha1_size = f.readline().strip().split(':')[-1].split(',')
        m = int(alpha1_size[0])
        n = int(alpha1_size[1])
        alpha1_array = np.array(f.readline().strip().split(','),dtype = np.float)
        self.setAlpha1(alpha1_array.reshape(m,n))
        
        alpha2_size = f.readline().strip().split(':')[-1].split(',')
        m = int(alpha2_size[0])
        n = int(alpha2_size[1])
        alpha2_array = np.array(f.readline().strip().split(','),dtype = np.float)
        self.setAlpha2(alpha2_array.reshape(m,n))
        
        beta1_size = f.readline().strip().split(':')[-1].split(',')
        m = int(beta1_size[0])
        n = int(beta1_size[1])
        beta1_array = np.array(f.readline().strip().split(','),dtype = np.float)
        self.setBeta1(beta1_array.reshape(m,n))
        
        beta2_size = f.readline().strip().split(':')[-1].split(',')
        m = int(beta2_size[0])
        n = int(beta2_size[1])
        beta2_array = np.array(f.readline().strip().split(','),dtype = np.float)
        self.setBeta2(beta2_array.reshape(m,n))
               
        f.readline()
        self.setTargetMean1(float(f.readline().strip()))
        
        f.readline()
        self.setTargetMean2(float(f.readline().strip()))
        
        # f.readline()
        # self.setTargetStd1(float(f.readline().strip()))
        
        # f.readline()
        # self.setTargetStd2(float(f.readline().strip()))
        
        f.readline()
        self.setTheta1(float(f.readline().strip()))
        
        f.readline()
        self.setTheta2(float(f.readline().strip()))
        
        f.readline()
        self.setEta1(np.array(f.readline().strip().split(','),dtype = np.float))
        
        f.readline()
        self.setEta2(np.array(f.readline().strip().split(','),dtype = np.float))
        
        f.readline()
        self.setSampleSize(int(f.readline().strip()))
        
        f.readline()
        self.setThetaGp1([float(i) for i in f.readline().split(',')])
        
        f.readline()
        self.setThetaGp2([float(i) for i in f.readline().split(',')])
        
        etaGp1_size =  f.readline().strip().split(':')[-1].split(',')
        m = int(etaGp1_size[0])
        n = int(etaGp1_size[1])
        etaGp1_array = np.array(f.readline().strip().split(','), dtype = np.float)
        self.setEtaGp1(etaGp1_array.reshape(m,n))
        
        etaGp2_size =  f.readline().strip().split(':')[-1].split(',')
        m = int(etaGp2_size[0])
        n = int(etaGp2_size[1])
        etaGp2_array = np.array(f.readline().strip().split(','), dtype = np.float)
        self.setEtaGp2(etaGp2_array.reshape(m,n))
        
        feature_name = self.getFeatureName()
        feature_type = self.getFeatureType()
        num_of_feature = len(feature_type)
        
        simTable1=[]
        simTable2=[]
        
        #theta1 = self.getTheta1()
        delta1 = np.exp(self.getEta1())
        
        #theta2 = self.getTheta2()
        delta2 = np.exp(self.getEta2())
        
        Idx_sr = self.getIdxSr()
        
        for k in range(0,num_of_feature):
            Type = feature_type[k]
            data_sr = Idx_sr[:,k]
            data_sr = data_sr.reshape((len(data_sr),1))
            
            if Type == 'IP':
                data = np.array([i for i in range(0,256)])
                data = np.transpose(data)
                data = data.reshape((len(data),1))
            elif Type == 'Port': 
                data = np.array([i for i in range(0, 65536)])
                data = np.transpose(data)
                data = data.reshape((len(data),1))
            elif Type == 'Categorical':
                data = np.array([i for i in range(0,2)])
                data = np.transpose(data)
                data = data.reshape((len(data),1))
                
            elif Type == 'Hour':
                data = np.array([i for i in range(0,24)])
                data = np.transpose(data)
                data = data.reshape((len(data),1))
            elif Type == 'Minute':
                data = np.array([i for i in range(0,60)])
                data = np.transpose(data)
                data = data.reshape((len(data),1))
            else:
                print "Error"
            
            D_k = NWOperation.nw_k_cplt(data, data_sr, [Type])
            
            K_1_k = NWOperation.kernel_gauss_nw_cplt(D_k, 1, [delta1[k]] )
            K_2_k = NWOperation.kernel_gauss_nw_cplt(D_k, 1, [delta2[k]])
            
            simTable1.append(K_1_k)
            simTable2.append(K_2_k)
        
        self.setSimTable1(simTable1)
        self.setSimTable2(simTable2)        
        f.close();
        
    def save(self, model_path): 
        f = open(model_path, 'wb')
        f.write('## TARGET_NAME:1,1\n')
        f.write(self.getTargetName() + '\n')
        f.write('##FEATURE_NAME:1,')
        f.write(str(len(self.getFeatureName())) + '\n')
        f.write(','.join(self.getFeatureName()) + '\n')
        f.write('##FEATURE_TYPE:1,')
        f.write(str(len(self.getFeatureType())) + '\n')
        f.write(','.join(self.getFeatureType()) + '\n')
        f.write('##FEATURE_WEIGHT:1,')
        f.write(str(len(self.getFeatureWeight())) + '\n')
        f.write(','.join([str(i) for i in self.getFeatureWeight()]) + '\n')
        f.write('## MAX_ITERATIONS: 1, 1\n')
        f.write(str(self.getMaxIter())+'\n')
        f.write('## RGLR: 1, 1\n')
        f.write(str(self.getRglr()) + '\n')
        f.write('## LEARNING_RATE: 1,1\n')
        f.write(str(self.getLearningRate()) + '\n')
        idxSr = self.getIdxSr()
        (m, n) = idxSr.shape
        f.write('##IDXSR:' + str(m) + ',' + str(n) + '\n')
        l = ''
        for a in range(m):
            for b in range(n):
                l += str(idxSr[a][b])
                l += ','
        f.write(l[:-1] + '\n')
        
        lambda1 = self.getLambda1()
        (m, n) = lambda1.shape
        f.write('## LAMBDA1:' + str(m) + ',' + str(n) + '\n')
        l = ''
        for a in range(m):
            for b in range(n):
                l += str(float(lambda1[a][b]))
                l += ', '
        f.write(l[:-2] + '\n')
        
        lambda2 = self.getLambda2()
        (m, n) = lambda2.shape
        f.write('## LAMBDA2:' + str(m) + ',' + str(n) + '\n')
        l = ''
        for a in range(m):
            for b in range(n):
                l += str(float(lambda2[a][b]))
                l += ', '
        f.write(l[:-2] + '\n')
        
        alpha1 = self.getAlpha1()
        (m, n) = alpha1.shape
        f.write('##ALPHA1:' + str(m) + ',' +str(n) + '\n')
        l = ''
        for a in range(m):
            for b in range(n):
                l += str(alpha1[a][b])
                l += ','
        f.write(l[:-1] + '\n')
        
        alpha2 = self.getAlpha2()
        (m, n) = alpha2.shape
        f.write('##ALPHA2:' + str(m) + ',' + str(n) + '\n')
        l = ''
        for a in range(m):
            for b in range(n):
                l += str(alpha2[a][b])
                l += ','
        f.write(l[:-1] + '\n')
        
        
        beta1 = self.getBeta1()
        (m, n) = beta1.shape
        f.write('##BETA1:' + str(m) + ',' +str(n) + '\n')
        l = ''
        for a in range(m):
            for b in range(n):
                l += str(beta1[a][b])
                l += ','
        f.write(l[:-1] + '\n')
        
        beta2 = self.getBeta2()
        (m, n) = beta2.shape
        f.write('##BETA2:' + str(m) + ',' + str(n) + '\n')
        l = ''
        for a in range(m):
            for b in range(n):
                l += str(beta2[a][b])
                l += ','
        f.write(l[:-1] + '\n')
        
        
        f.write('##TARGET_MEAN1:1,1\n')
        f.write(str(self.getTargetMean1()) + '\n')
        f.write('##TARGET_MEAN2:1,1\n')
        f.write(str(self.getTargetMean2()) + '\n')
        # f.write('##TARGET_STD1:1,1\n')
        # f.write(str(self.getTargetStd1()) + '\n')
        # f.write('##TARGET_STD2:1,1\n')
        # f.write(str(self.getTargetStd2()) + '\n')
        
        f.write('##THETA1:1,1\n')
        f.write(str(self.getTheta1()) + '\n')
        f.write('THETA2:1,1\n')
        f.write(str(self.getTheta2()) + '\n')
        eta1 = self.getEta1()
        eta_gp1_len = len(eta1)
        f.write('##ETA1:1,' + str(eta_gp1_len) + '\n')
        l = ''
        for a in range(eta_gp1_len):
            l += str(eta1[a])
            l += ','
        f.write(l[:-1] + '\n')
        
        eta2 = self.getEta2()
        eta_gp2_len = len(eta2)
        
        f.write('##ETA2:1,' + str(eta_gp2_len) + '\n')
        l = ''
        for a in range(eta_gp2_len):
            l += str(eta2[a])
            l += ','
        f.write(l[:-1] + '\n')

        f.write('## SAMPLE_SIZE:1,1\n')
        f.write(str(self.getSampleSize()) + '\n')
        
        Theta_gp1 = self.getThetaGp1()
        ThetaGp1Len = len(Theta_gp1)
        f.write('##THETA_GP1:1,')
        f.write(str(ThetaGp1Len) + '\n')
        l = ''
        for a in Theta_gp1:
            l += str(a)
            l += ','
        f.write(l[:-1] + '\n')
        
        Theta_gp2 = self.getThetaGp2()
        ThetaGp2Len = len(Theta_gp2)
        f.write('##THETA_GP2:1,')
        f.write(str(ThetaGp2Len) + '\n')
        l = ''
        for a in Theta_gp2:
            l += str(a)
            l +=','
        f.write(l[:-1]+ '\n')
        
        Eta_gp1 = self.getEtaGp1()
        (m, n) = Eta_gp1.shape
        f.write('##ETA_GP1:' + str(m) + ',' + str(n) + '\n')
        l = ''
        for a in range(m):
            for b in range(n):
                l += str(Eta_gp1[a][b])
                l += ','
        l = l[:-1] + ']'
        f.write(l[:-1] + '\n')
        Eta_gp2 = self.getEtaGp2()
        (m, n) = Eta_gp2.shape
        f.write('##ETA_GP2:' + str(m) + ',' +str(n) + '\n')
        l = ''
        for a in range(m):
            for b in range(n):
                l += str(Eta_gp2[a][b])
                l += ','
        l = l[:-1] + ']'
        f.write(l[:-1] + '\n')

    def setFeatureName(self, name):
        self.featureName = name
        
    def setTargetName(self,name):
        self.targetName = name
        
    def setFeatureType(self,feat_type):
        self.featureType = feat_type
        
    def setFeatureWeight(self,weight):
        self.featureWeight = weight
        
    def setMaxIter(self,iter):
        self.maxIter = iter
        
    def setRglr(self,r):
        self.rglr = r
        
    def setLearningRate(self,rate):
        self.learningRate = rate
     
    def setIdxSr(self,data):
        self.idxSr = data
        
    def setLambda1(self, Lambda):
        self.lambda1 = Lambda
     
    def setLambda2(self,Lambda):
        self.lambda2 = Lambda
    
    def setAlpha1(self,alpha):
        self.alpha1 = alpha
    
    def setAlpha2(self, alpha):
        self.alpha2 = alpha
        
    def setBeta1(self, beta):
        self.beta1 = beta
    
    def setBeta2(self, beta):
        self.beta2 = beta    

    def setTargetMean1(self,mean):
        self.targetMean1 = mean
   
    def setTargetMean2(self,mean):
        self.targetMean2 = mean
   
    # def setTargetStd1(self,Std):
        # self.targetStd1 = Std

    # def setTargetStd2(self, Std):
        # self.targetStd2 =Std

    def setTheta1(self,theta):
        self.theta1 = theta
   
    def setTheta2(self,theta):
        self.theta2 = theta
    
    def setEta1(self, eta):
        self.eta1 = eta
        
    def setEta2(self,eta):
        self.eta2 = eta
        
    def setSampleSize(self, sample):
        self.sampleSize = sample
        
    def setThetaGp1(self, thetaGp):
        self.thetaGp1 = thetaGp
        
    def setThetaGp2(self, thetaGp):
        self.thetaGp2 = thetaGp
    
    def setEtaGp1(self,etaGp):
        self.etaGp1 = etaGp
    
    def setEtaGp2(self, etaGp):
        self.etaGp2 = etaGp
        
    def setSimTable1(self,simTable1):  
        self.simTable1 = simTable1
        
    def setSimTable2(self, simTable2):
        self.simTable2 = simTable2

    def getFeatureName(self):
        return self.featureName
    
    def getTargetName(self):
        return self.targetName
    
    def getFeatureType(self):
        return self.featureType
        
    def getFeatureWeight(self):
        return self.featureWeight

    def getMaxIter(self):
        return self.maxIter
    
    def getRglr(self):
        return self.rglr
    
    def getLearningRate(self):
        return self.learningRate
        
    def getIdxSr(self):
        return self.idxSr
    
    def getLambda1(self):
        return self.lambda1
        
    def getLambda2(self):
        return self.lambda2
        
    def getAlpha1(self):
        return self.alpha1
        
    def getAlpha2(self):
        return self.alpha2
        
    def getBeta1(self):
        return self.beta1
        
    def getBeta2(self):
        return self.beta2
    
    def getTargetMean1(self):
        return self.targetMean1
    
    def getTargetMean2(self):
        return self.targetMean2
        
    #def getTargetStd1(self):
        #return self.targetStd1
    
    # def getTargetStd2(self):
        # return self.targetStd2
    
    def getTheta1(self):
        return self.theta1
    
    def getTheta2(self):
        return self.theta2
        
    def getEta1(self):
        return self.eta1
     
    def getEta2(self):
        return self.eta2
        
    def getSampleSize(self):
        return self.sampleSize
        
    def getThetaGp1(self):
        return self.thetaGp1
        
    def getThetaGp2(self):
        return self.thetaGp2
        
    def getEtaGp1(self):
        return self.etaGp1
     
    def getEtaGp2(self):
        return self.etaGp2
    
    def getSimTable1(self):
        return self.simTable1
    
    def getSimTable2(self):
        return self.simTable2
        
class NWOperation:
    @staticmethod
    def delta_port(x):
        if x >= 0 and x <= 1023:
           delta=0
        elif x >= 1024 and x <= 49151:
           delta=1
        else: 
           delta=2
        return delta
        
    @staticmethod
    def dist_port(x1,x2):
        delta_x1 =  NWOperation.delta_port(x1)
        delta_x2 = NWOperation.delta_port(x2)
        if x1 == x2:
           dist = float(0)
        elif delta_x1 == delta_x2:
           dist = float(1)
        elif (delta_x1 == 0 or delta_x1 ==1) and (delta_x2 == 0 or delta_x2 == 1):
           dist = float(2) 
        else:
            dist = float (4) 
        return dist
  
    @staticmethod
    def convert_to_bit(ip):
        ret = ''
        v = int (ip)
        while v > 0:
           ret += str(v % 2)
           v = v / 2
        if len(ret) < 8 :
           ret += '0' * (8 - len(ret))
 
        return ret[::-1]

    @staticmethod
    def is_binary(num):
        b= str(num)
        for  i in b:
            if i == '0' or i == '1':
               continue
            return False
        if len(b) == 8:
            return True
        else:
            return False

    @staticmethod
    def dist_IP(ip1, ip2):
        L = 0
        IP1 = int (ip1)
        IP2 = int (ip2)
        if not NWOperation.is_binary(IP1):
           IP1  = NWOperation.convert_to_bit(IP1)
        
        if not NWOperation.is_binary(IP2):
           IP2 = NWOperation.convert_to_bit(IP2)

        Len = min(len(IP1), len(IP2))
  
        for k in range(0,Len):
            if IP1[k] != IP2[k]:
               break
            L = L+1
        d = np.log(float(Len+1)/float(L+1))

        return d

    @staticmethod 
    def dist_protocol(x1,x2):
        if x1== x2:
           dist = 0
        else: 
           dist = 1
        return dist
    
    @staticmethod
    def kernel_gauss_nw_cplt(dist,theta,delta):
        Len = len(dist)    
       
        m,n = dist[0].shape

   
        y = np.zeros((m,n), dtype = float)
        
        for i in range(0, Len):
            y = y + np.exp(-dist[i]/2/delta[i])/Len

        y = theta * (y ** Len)
        return y

    @staticmethod
    def nw_k_cplt(data, data_sr, feature_type):
        
        len1 = len(data)
        len2 = len(data_sr)

        xx1 = np.transpose(data)
        xx2 = np.transpose(data_sr)

        temp = []
        for x in xx1: 
            temp.append(x.tolist()) 
        xx1 = temp 
    
        temp = []
        for x in xx2: 
            temp.append(x.tolist()) 
        xx2 = temp
   
        num_of_feature = len(feature_type)
        K = []
        for i in range(0, num_of_feature):
            K_k = np.zeros((len1, len2), dtype = float)
            K.append(K_k)
        dist_x1_x2 = 0.0

        for i in range(0, len1):
            for j in range(0,len2):
                for k in range(0, num_of_feature):
                    Type = feature_type[k]
                    x1 = xx1[k]
                    x2 = xx2[k]
                    if Type == 'numeric':
                       dist_x1_x2 = (x1[i]- x2[j])**2/np.abs(x1[i]*x2[j])
                    elif Type == 'IP':
                       dist_x1_x2 = (NWOperation.dist_IP(x1[i],x2[j]))**2
                    elif Type == 'Port':
                        dist_x1_x2 = (NWOperation.dist_port(x1[i],x2[j]))**2
                    elif Type == 'Categorical':
                        dist_x1_x2 = (NWOperation.dist_protocol(x1[i],x2[j]))**2
                    else: 
                        dist_x1_x2 = 0.0

                    K[k][i][j] = dist_x1_x2

        return K
        
    @staticmethod
    def add_new_info(K, y_new, Lambda, Alpha, Beta, target_mean, Str):
         
         f_fgt = 0.999
         
         target_mean = float(Str)/float(Str+1) * target_mean  + 1/float(Str)*y_new
         Str = Str + 1
         
         Alpha = f_fgt * Alpha + y_new * np.dot(Lambda, K)
         Beta = f_fgt * Beta + np.dot(Lambda, K)
    
         return Alpha, Beta, target_mean, Str
         
         
if __name__ == '__main__':
    input = 'D:\\Project MIND\\packet_trace_data\\univ_1_4.csv'
    target_name = 'Payload_Bytes'
    feature_name = 'Server_IP_1:Server_IP_2:Server_IP_3:Server_IP_4:Client_IP_1:Client_IP_2:Client_IP_3:Client_IP_4:Server_Port'
    feature_type = 'IP:IP:IP:IP:IP:IP:IP:IP:Port'
    feature_weight = '0.2:0.2:0.2'
    start = 0
    length =600
    print 'training stage...'
    nw = NWFlowEstimator()
    model = nw.train(input, target_name, feature_name, feature_type, feature_weight, start, length)
    #print 'batch test stage'
    model2 = NWModel()
    model2.load('NW_model.txt')
    #model2.predict(input, 400, 100)
    print 'single test stage ...'
    singleTestSuite = ["123:34:56:0:234:123:3:20:344:30"]
    # print "Before optimization"
    # tic = time.time()
    # for i in range(0,1000):
    a=model2.predictSingle(singleTestSuite[0],0)
    print a 
    # toc = time.time()
    # print "The time for a single flow prediction is", float(toc-tic)/1000
    # print "The predicted flow size is", a
    # print "After optimization"
    # tic = time.time()
    # for i in range(0,1000):
        # a = model2.predictSingle_2(singleTestSuite[0],0)
    # toc = time.time()
    # print "The time for predicting one flow is", float(toc-tic)/1000
    # print "The predicted flow size is", a  
    # print "Online update"
    # tic = time.time()
    # for i in range(0,1000):
        # model2.update_online(singleTestSuite[0], 10000)
    # toc = time.time()
    # print "The time for one model update is", float(toc-tic)/1000
    #print 'cross validate stage...'
    #gpr.cross_validate(input, target_name, feature_name, feature_type, feature_weight, 0, 500, 0.8)
    #print 'update stage...'
    #new_model = gpr.update(input, 'model.txt', 400, 100, False)
    print "Testing a university data center data >>>>>>>"
    df = pd.read_csv(input)
    data_len = len(df.loc[:,target_name])
    
    thrsh_true = 1e+4
    real_label = np.zeros((data_len - length - start), dtype = np.integer)
    label_test = np.zeros((data_len - length - start), dtype = np.integer)
    TP = 0
    FP = 0
    
    W_mice = 0
    for i in range(start+length, data_len):
        sample = df.loc[i,model2.getFeatureName()]
        test_data = ''
        for k in range(0, len(model2.getFeatureName())):
            test_data += str(sample[k]) + ':'
        test_data = test_data [:-1]
        fs_pred = model2.predictSingle_2(test_data, 0)
        fs_real = df.loc[i, target_name]
        model2.update_online(test_data, fs_real)
        #print "The real flow size is", fs_real
        #print "The predicted flow size is", fs_pred
        
        if fs_real >= thrsh_true:
           real_label[i-start-length] = 1
        if fs_pred >= thrsh_true:
           label_test[i-start-length] = 1
        if real_label[i-start-length]==1 and label_test[i-start-length] ==1:
           TP = TP + 1
        if real_label[i-start-length]==0:
           W_mice = W_mice + (1-float(fs_real)/thrsh_true)
        if real_label[i-start-length] == 0 and label_test[i-start-length]==1:
           FP = FP + (1-float(fs_real)/thrsh_true)
        
    TPR = float(TP)/float(sum(real_label))
    FNR = 1-TPR
        
    FPR = float(FP)/W_mice
    TNR = 1 -FPR
        
    print "The TPR is", TPR
    print "The FNR is", FNR
    print "The FPR is", FPR
    print "The TNR is", TNR    
    