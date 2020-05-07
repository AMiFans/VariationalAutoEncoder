import tensorflow as tf
import matplotlib.pyplot as plt


class Sampling(tf.keras.layers.Layer):
    """重参数化，使用 (z_mean, z_log_var) 采样 z"""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(tf.keras.layers.Layer):
    """"编码器，将输入inputs映射为三元组 (z_mean, z_log_var, z)."""
    def __init__(self,
                 latent_dim=32,
                 name='encoder',
                 **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu')
        self.flat = tf.keras.layers.Flatten()
        # No activation
        self.mean = tf.keras.layers.Dense(latent_dim)
        self.log_var = tf.keras.layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flat(x)
        z_mean = self.mean(x)
        z_log_var = self.log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(tf.keras.layers.Layer):
    """"解码器"""
    def __init__(self,
                 name='decoder',
                 **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense = tf.keras.layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu)
        self.reshape = tf.keras.layers.Reshape(target_shape=(7, 7, 32))
        self.convt1 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu')
        self.convt2 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation='relu')
        # No activation
        self.convt3 = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=(1, 1), padding="SAME")

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.reshape(x)
        x = self.convt1(x)
        x = self.convt2(x)
        x = self.convt3(x)
        return x


class VariationalAutoEncoder(tf.keras.Model):
    """将编码器和解码器组合为端到端模型以进行训练"""
    def __init__(self,
                 latent_dim=32,
                 name='autoencoder',
                 **kwargs):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder()

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = - 0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.add_loss(0.1*kl_loss)
        return reconstructed


epochs = 5
num_examples_to_generate = 100
latent_dim = 128

# 构建模型
vae = VariationalAutoEncoder(latent_dim=latent_dim)

# 加载数据
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()  # [60000, 28, 28]

# 将图片标准化到 [-1, 1] 区间内
x_train, x_test = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32'), x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
x_train, x_test = (x_train - 127.5) / 127.5, (x_test - 127.5) / 127.5

train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(buffer_size=1024).batch(128)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
mse_loss_fn = tf.keras.losses.MeanSquaredError()
loss_metric = tf.keras.metrics.Mean()


# 保持随机向量恒定以进行生成（预测），以便更易于看到改进。
random_vector_for_generation = tf.random.normal(shape=[num_examples_to_generate, latent_dim])
test_input = random_vector_for_generation


def generate_and_save_images(model, epoch, test_input):
    predictions = model.decoder(test_input)
    plt.figure(figsize=(10, 10))

    for i in range(predictions.shape[0]):
        image = predictions
        plt.subplot(10, 10, i+1)
        plt.imshow(image[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.savefig('./save_images/vae_cnn_image_at_epoch_{:04d}.png'.format(epoch))
    #plt.show()


# 迭代5次.
for epoch in range(epochs):
    print('Start of epoch %d' % (epoch,))

    # 遍历数据集的所有batch
    for step, x_batch_train in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            reconstructed = vae(x_batch_train)
            # 计算重构损失
            loss = mse_loss_fn(x_batch_train, reconstructed)
            loss += sum(vae.losses)  # 增加 KLD 正则损失

        grads = tape.gradient(loss, vae.trainable_weights)
        optimizer.apply_gradients(zip(grads, vae.trainable_weights))

        loss_metric(loss)

        if step % 100 == 0:
            print('step %s: mean loss = %s' % (step, loss_metric.result().numpy()))
    if epoch % 10 == 0:
        generate_and_save_images(vae, epoch, test_input)

vae.save_weights('save_weights/vae_cnn_weights')
