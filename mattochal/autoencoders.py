cnn_models = {"shallow_cnn_4lyr_3ch_cifar":
          lambda input_shape: 
          Sequential([
              Conv2D(3, (3, 3), activation='relu', padding='same', input_shape=input_shape), 
              MaxPooling2D(pool_size=2),
              UpSampling2D((2, 2)),
              Conv2D(3, (3, 3), activation='sigmoid', padding='same')
          ]),
          
          "shallow_cnn_8ch_mnist": 
          lambda input_shape: 
          Sequential([
              Conv2D(8, (3, 3), activation='relu', padding='same', input_shape=input_shape), 
              MaxPooling2D(pool_size=2),
              Conv2D(1, (3, 3), activation='relu', padding='same'),
              Conv2D(8, (3, 3), activation='relu', padding='same'), 
              UpSampling2D((2, 2)),
              Conv2D(1, (3, 3), activation='sigmoid', padding='same')
          ]),
    
          "shallow_cnn_8ch_cifar": 
          lambda input_shape: 
          Sequential([
              Conv2D(8*3, (3, 3), activation='relu', padding='same', input_shape=input_shape), 
              MaxPooling2D(pool_size=2),
              Conv2D(3, (3, 3), activation='relu', padding='same'),
              Conv2D(8*3, (3, 3), activation='relu', padding='same'), 
              UpSampling2D((2, 2)),
              Conv2D(3, (3, 3), activation='sigmoid', padding='same')
          ]),
          
          "shallow_cnn_8ch_mnist": 
          lambda input_shape: 
          Sequential([
              Conv2D(8, (3, 3), activation='relu', padding='same', input_shape=input_shape), 
              MaxPooling2D(pool_size=2),
              Conv2D(1, (3, 3), activation='relu', padding='same'),
              Conv2D(8, (3, 3), activation='relu', padding='same'), 
              UpSampling2D((2, 2)),
              Conv2D(1, (3, 3), activation='sigmoid', padding='same')
          ]),
          
          "shallow_cnn_16ch_cifar": 
          lambda input_shape: 
          Sequential([
              Conv2D(16*3, (3, 3), activation='relu', padding='same', input_shape=input_shape), 
              MaxPooling2D(pool_size=2),
              Conv2D(3, (3, 3), activation='relu', padding='same'),
              Conv2D(16*3, (3, 3), activation='relu', padding='same'), 
              UpSampling2D((2, 2)),
              Conv2D(3, (3, 3), activation='sigmoid', padding='same')
          ]),
          
          "shallow_cnn_16ch_mnist": 
          lambda input_shape: 
          Sequential([
              Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shape), 
              MaxPooling2D(pool_size=2),
              Conv2D(3, (3, 3), activation='relu', padding='same'),
              Conv2D(16, (3, 3), activation='relu', padding='same'), 
              UpSampling2D((2, 2)),
              Conv2D(3, (3, 3), activation='sigmoid', padding='same')
          ])
         }


def model_train(x_train, y_train, x_test, y_test, epochs=5, batch_size=128, retrain=False, model_name="shallow_cnn_16ch_mnist", model_fn):
    model_filename = model_name + ".h5"
    
    if os.path.isfile(model_filename) and not retrain:
        model = load_model(model_filename)
        model.summary()
    else:
        input_image_dim= np.shape(x_train[0])
        model = model_fn(input_image_dim)
        
        adam = Adam(lr=0.0003)
        model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
        print(srcnn_model.summary())
        
        checkpoint = ModelCheckpoint("SRCNN_check_longer.h5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min')
        callbacks_list = [checkpoint]
        print("Training")
        model.fit(x_train, y_train,
                     epochs=epochs,
                     batch_size=batch_size,
                     shuffle=True,
                     callbacks=callbacks_list,
                     validation_data=(x_test, y_test))
        model.save(model_filename)
    return model


for model_name, model_fn in cnn_models.items():
    ts = time.time()
    if "mnist" in model_name:
        model = model_train(
            x_train = x_train_mnist,
            y_train = x_train_mnist,
            x_test = x_test_mnist,
            y_test = x_test_mnist,
            model_fn = model_fn
        )
    elif "cifar" in model_name:
        model = model_train(
            x_train = x_train_cifar,
            y_train = x_train_cifar,
            x_test = x_test_cifar,
            y_test = x_test_cifar,
            model_fn = model_fn
        )
    te = time.time()
    print("Time taken: {:.2f}".format(te-ts))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    