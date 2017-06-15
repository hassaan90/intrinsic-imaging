class Histories(keras.callbacks.Callback):

    def __init__(self):
        super(Histories, self).__init__()
        self.directory = '../output2/prediction'
        self.count = 0
        #self.count1 = 0
        self.nepoch = 0
    
	def on_batch_begin(self, batch, logs={}):
            return

	def on_batch_end(self, batch, logs={}):
            
            self.losses.append(logs.get('loss'))
            print self.losses[0]
            if self.count >49:
                self.count = 0
            y_pred = self.model.predict(self.model.validation_data[0][self.count:self.count+1,:,:,:])
            #print y_pred.shape
            #print type(y_pred)
            y_pred = (y_pred - np.min(y_pred))
            
            y_pred = y_pred / np.max(y_pred)
            y_pred = y_pred[0]
            y_gt = model.validation_data[1][self.count]
            y_pred = y_pred.transpose(1,2,0)
            y_gt = y_gt.transpose(1,2,0)
            result = np.concatenate((y_gt,y_pred ), axis=1)
            print np.max(y_pred), np.min(y_pred), np.max(y_gt), np.min(y_gt)
            #plt.imshow(result)
            #plt.show()
            #pause(0.01)
            oname = os.path.join(self.directory, str(self.count)+'_'+str(self.nepoch)+'_'+str(batch)+'.jpg')
            print oname
            time.sleep(5)
            self.count += 1
            cv2.imwrite(oname, result*255)

            #plt.imshow(result)
            #cv2.imwrite('gt.jpg'+str(Histories.count)+'.jpg', y_gt)
            return
    	def on_epoch_begin(self, epoch, logs={}):
            self.nepoch = epoch
            return

    	def on_train_begin(self, logs={}):
            self.losses = []
            return