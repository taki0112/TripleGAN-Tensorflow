import cifar10
from ops import *
from utils import *
import time

class TripleGAN(object) :
    def __init__(self, sess, epoch, batch_size, unlabel_batch_size, z_dim, dataset_name, n, checkpoint_dir, result_dir, log_dir):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.epoch = epoch
        self.batch_size = batch_size
        self.unlabelled_batch_size = unlabel_batch_size
        self.test_batch_size = 1000
        self.model_name = "TripleGAN"     # name for checkpoint
        if self.dataset_name == 'cifar10' :
            self.input_height = 32
            self.input_width = 32
            self.output_height = 32
            self.output_width = 32

            self.z_dim = z_dim
            self.y_dim = 10
            self.c_dim = 3

            self.learning_rate = 3e-4 # 3e-4, 1e-3
            self.cla_learning_rate = 3e-3 # 3e-3, 1e-2 ?
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8
            self.alpha = 0.5
            self.init_alpha_p = 0.0 # 0.1, 0.03
            self.apply_alpha_p = 0.1
            self.apply_epoch = 200 # 200, 300

            self.sample_num = 64
            self.visual_num = 100
            self.len_discrete_code = 10

            self.data_X, self.data_y, self.unlabelled_X, self.unlabelled_y, self.test_X, self.test_y = cifar10.prepare_data(n) # trainX, trainY, testX, testY

            self.num_batches = len(self.data_X) // self.batch_size

        else :
            raise NotImplementedError

    def discriminator(self, x, y_, scope='discriminator', is_training=True, reuse=False):
        with tf.variable_scope(scope, reuse=reuse) :
            x = dropout(x, rate=0.2, is_training=is_training)
            y = tf.reshape(y_, [-1, 1, 1, self.y_dim])
            x = conv_concat(x,y)

            x = lrelu(conv_layer(x, filter_size=32, kernel=[3,3], layer_name=scope+'_conv1'))
            x = conv_concat(x,y)
            x = lrelu(conv_layer(x, filter_size=32, kernel=[3,3], stride=2, layer_name=scope+'_conv2'))
            x = dropout(x, rate=0.2, is_training=is_training)
            x = conv_concat(x,y)

            x = lrelu(conv_layer(x, filter_size=64, kernel=[3,3], layer_name=scope+'_conv3'))
            x = conv_concat(x,y)
            x = lrelu(conv_layer(x, filter_size=64, kernel=[3,3], stride=2, layer_name=scope+'_conv4'))
            x = dropout(x, rate=0.2, is_training=is_training)
            x = conv_concat(x,y)

            x = lrelu(conv_layer(x, filter_size=128, kernel=[3,3], layer_name=scope+'_conv5'))
            x = conv_concat(x,y)
            x = lrelu(conv_layer(x, filter_size=128, kernel=[3,3], layer_name=scope+'_conv6'))
            x = conv_concat(x,y)

            x = Global_Average_Pooling(x)
            x = flatten(x)
            x = concat([x,y_]) # mlp_concat

            x_logit = linear(x, unit=1, layer_name=scope+'_linear1')
            out = sigmoid(x_logit)

            return out, x_logit, x

    def generator(self, z, y, scope='generator', is_training=True, reuse=False):
        with tf.variable_scope(scope, reuse=reuse) :

            x = concat([z, y]) # mlp_concat

            x = relu(linear(x, unit=512*4*4, layer_name=scope+'_linear1'))
            x = batch_norm(x, is_training=is_training, scope=scope+'_batch1')

            x = tf.reshape(x, shape=[-1, 4, 4, 512]) # -1이 아니라 self.batch_size인가 ?
            y = tf.reshape(y, [-1, 1, 1, self.y_dim])
            x = conv_concat(x,y)

            x = relu(deconv_layer(x, filter_size=256, kernel=[5,5], stride=2, layer_name=scope+'_deconv1'))
            x = batch_norm(x, is_training=is_training, scope=scope+'_batch2')
            x = conv_concat(x,y)

            x = relu(deconv_layer(x, filter_size=128, kernel=[5,5], stride=2, layer_name=scope+'_deconv2'))
            x = batch_norm(x, is_training=is_training, scope=scope+'_batch3')
            x = conv_concat(x,y)

            x = tanh(deconv_layer(x, filter_size=3, kernel=[5,5], stride=2, wn=True, layer_name=scope+'deconv3'))

            return x
    def classifier(self, x, scope='classifier', is_training=True, reuse=False):
        with tf.variable_scope(scope, reuse=reuse) :
            x = gaussian_noise_layer(x) # default = 0.15
            x = lrelu(conv_layer(x, filter_size=128, kernel=[3,3], layer_name=scope+'_conv1'))
            x = lrelu(conv_layer(x, filter_size=128, kernel=[3,3], layer_name=scope+'_conv2'))
            x = lrelu(conv_layer(x, filter_size=128, kernel=[3,3], layer_name=scope+'_conv3'))

            x = max_pooling(x, kernel=[2,2], stride=2)
            x = dropout(x, rate=0.5, is_training=is_training)

            x = lrelu(conv_layer(x, filter_size=256, kernel=[3,3], layer_name=scope+'_conv4'))
            x = lrelu(conv_layer(x, filter_size=256, kernel=[3,3], layer_name=scope+'_conv5'))
            x = lrelu(conv_layer(x, filter_size=256, kernel=[3,3], layer_name=scope+'_conv6'))

            x = max_pooling(x, kernel=[2,2], stride=2)
            x = dropout(x, rate=0.5, is_training=is_training)

            x = lrelu(conv_layer(x, filter_size=512, kernel=[3,3], layer_name=scope+'_conv7'))
            x = nin(x, unit=256, layer_name=scope+'_nin1')
            x = nin(x, unit=128, layer_name=scope+'_nin2')

            x = Global_Average_Pooling(x)
            x = flatten(x)
            x = linear(x, unit=10, layer_name=scope+'_linear1')
            return x

    def build_model(self):
        image_dims = [self.input_height, self.input_width, self.c_dim]
        bs = self.batch_size
        unlabel_bs = self.unlabelled_batch_size
        test_bs = self.test_batch_size
        alpha = self.alpha
        # alpha_p = self.alpha_p
        self.alpha_p = tf.placeholder(tf.float32, name='alpha_p')
        """ Graph Input """
        # images
        self.inputs = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')
        self.unlabelled_inputs = tf.placeholder(tf.float32, [unlabel_bs] + image_dims, name='unlabelled_images')
        self.test_inputs = tf.placeholder(tf.float32, [test_bs] + image_dims, name='test_images')

        # labels
        self.y = tf.placeholder(tf.float32, [bs, self.y_dim], name='y')
        self.test_label = tf.placeholder(tf.float32, [test_bs, self.y_dim], name='test_label')
        self.visual_y = tf.placeholder(tf.float32, [self.visual_num, self.y_dim], name='visual_y')

        # noises
        self.z = tf.placeholder(tf.float32, [bs, self.z_dim], name='z')
        self.visual_z = tf.placeholder(tf.float32, [self.visual_num, self.z_dim], name='visual_z')

        """ Loss Function """
        # A Game with Three Players

        # output of D for real images
        D_real, D_real_logits, _ = self.discriminator(self.inputs, self.y, is_training=True, reuse=False)

        # output of D for fake images
        G = self.generator(self.z, self.y, is_training=True, reuse=False)
        D_fake, D_fake_logits, _ = self.discriminator(G, self.y, is_training=True, reuse=True)

        # output of C for real images
        C_real_logits = self.classifier(self.inputs, is_training=True, reuse=False) # 실제 이미지
        R_L = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=C_real_logits)) # classify error 계산

        # output of D for unlabelled images
        Y_c = self.classifier(self.unlabelled_inputs, is_training=True, reuse=True) # 어떤 이미지를 넣어야하는거지 ? 걍 같은 input 이미지 ?
        D_cla, D_cla_logits, _ = self.discriminator(self.unlabelled_inputs, Y_c, is_training=True, reuse=True)  # class를 뱉어냄

        # output of C for fake images
        C_fake_logits = self.classifier(G, is_training=True, reuse=True) # 생성된 이미지를 classify
        R_P = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=C_fake_logits)) # 생성된 이미지의 라벨에 대한 에러 계산

        #

        # get loss for discriminator
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real))) # 진짜 이미지
        d_loss_fake = (1-alpha)*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake))) # 가짜 이미지
        d_loss_cla = alpha*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_cla_logits, labels=tf.zeros_like(D_cla))) # 진짜이미지인데 classify
        self.d_loss = d_loss_real + d_loss_fake + d_loss_cla

        # get loss for generator
        self.g_loss = (1-alpha)*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake))) # 가짜이미지

        # test loss for classify
        test_Y = self.classifier(self.test_inputs, is_training=False, reuse=True)
        correct_prediction = tf.equal(tf.argmax(test_Y, 1), tf.argmax(self.test_label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # get loss for classify
        max_c = tf.cast(tf.argmax(Y_c, axis=1), tf.float32)
        c_loss_dis = tf.reduce_mean(max_c * tf.nn.softmax_cross_entropy_with_logits(logits=D_cla_logits, labels=tf.ones_like(D_cla)))
        self.c_loss = alpha * c_loss_dis + R_L + self.alpha_p*R_P

        """ Training """

        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        g_vars = [var for var in t_vars if 'generator' in var.name]
        c_vars = [var for var in t_vars if 'classifier' in var.name]

        for var in t_vars: print(var.name)
        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2=self.beta2,epsilon=self.epsilon).minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1, beta2=self.beta2,epsilon=self.epsilon).minimize(self.g_loss, var_list=g_vars)
            self.c_optim = tf.train.AdamOptimizer(self.cla_learning_rate, beta1=self.beta1, beta2=self.beta2,epsilon=self.epsilon).minimize(self.c_loss, var_list=c_vars)


            """
            with tf.name_scope("ADAM") :
                batch = tf.Variable(0)

                learning_rate = tf.train.exponential_decay(
                    learning_rate=self.learning_rate, # Base learning rate
                    global_step=batch * self.batch_size, # Current index into the dataset
                    decay_steps=len(self.data_X), # Decay step
                    decay_rate=0.995, # Decay rate
                    staircase=True
                )
                self.d_optim = tf.train.AdamOptimizer(learning_rate, beta1=self.beta1, beta2=self.beta2, epsilon=self.epsilon).minimize(self.d_loss)
                self.g_optim = tf.train.AdamOptimizer(learning_rate, beta1=self.beta1, beta2=self.beta2, epsilon=self.epsilon).minimize(self.g_loss)
                self.c_optim = tf.train.AdamOptimizer(learning_rate, beta1=self.beta1, beta2=self.beta2, epsilon=self.epsilon).minimize(self.c_loss)
            """


        """" Testing """
        # for test
        self.fake_images = self.generator(self.visual_z, self.visual_y, is_training=False, reuse=True)

        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_cla_sum = tf.summary.scalar("d_loss_cla", d_loss_cla)

        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        c_loss_sum = tf.summary.scalar("c_loss", self.c_loss)



        # final summary operations
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])
        self.c_sum = tf.summary.merge([d_loss_cla_sum, c_loss_sum])


    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
        self.test_codes = self.data_y[0:self.batch_size]

        # saver to save model
        self.saver = tf.train.Saver()

        # summary writer
        self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches)
            start_batch_id = checkpoint_counter - start_epoch * self.num_batches
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        # loop for epoch
        start_time = time.time()
        for epoch in range(start_epoch, self.epoch):
            if epoch >= self.apply_epoch :
                alpha_p = self.apply_alpha_p
            else :
                alpha_p = self.init_alpha_p
            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                batch_images = self.data_X[idx * self.batch_size : (idx + 1) * self.batch_size]
                batch_codes = self.data_y[idx * self.batch_size : (idx + 1) * self.batch_size]

                # batch_unlabelled_images = np.asarray(random.sample(list(self.unlabelled_X), self.unlabelled_batch_size))
                batch_unlabelled_images = self.unlabelled_X[idx * self.unlabelled_batch_size : (idx + 1) * self.unlabelled_batch_size]
                # batch_unlabelled_images = random.sample(self.unlabelled_X, self.batch_size)
                # batch_unlabelled_codes = random.sample(self.unlabelled_y, self.batch_size)

                batch_z = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))

                # update D network
                _, summary_str, d_loss = self.sess.run([self.d_optim, self.d_sum, self.d_loss],
                                                       feed_dict={self.inputs: batch_images, self.y: batch_codes,
                                                                  self.unlabelled_inputs: batch_unlabelled_images,
                                                                  self.z: batch_z, self.alpha_p: alpha_p})
                self.writer.add_summary(summary_str, counter)

                # update G network
                _, summary_str_g, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss],
                    feed_dict={self.inputs: batch_images, self.y: batch_codes,
                               self.unlabelled_inputs: batch_unlabelled_images,
                               self.z: batch_z, self.alpha_p: alpha_p})
                self.writer.add_summary(summary_str_g, counter)

                # update C network
                _, summary_str_c, c_loss = self.sess.run([self.c_optim, self.c_sum, self.c_loss],
                    feed_dict={self.inputs: batch_images, self.y: batch_codes,
                               self.unlabelled_inputs: batch_unlabelled_images,
                               self.z: batch_z, self.alpha_p: alpha_p})
                self.writer.add_summary(summary_str_c, counter)

                # display training status
                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, c_loss: %.8f" \
                      % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss, c_loss))

                # save training results for every 300 steps
                """
                if np.mod(counter, 300) == 0:
                    samples = self.sess.run(self.fake_images,
                                            feed_dict={self.z: self.sample_z, self.y: self.test_codes})
                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                './' + check_folder(
                                    self.result_dir + '/' + self.model_dir) + '/' + self.model_name + '_train_{:02d}_{:04d}.png'.format(
                                    epoch, idx))
                """

            # classifier test
            test_acc = 0.0

            for idx in range(10) :
                test_batch_x = self.test_X[idx * self.test_batch_size : (idx+1) * self.test_batch_size]
                test_batch_y = self.test_y[idx * self.test_batch_size : (idx+1) * self.test_batch_size]

                acc_ = self.sess.run(self.accuracy, feed_dict={
                    self.test_inputs: test_batch_x,
                    self.test_label: test_batch_y
                })

                test_acc += acc_
            test_acc /= 10

            summary_test = tf.Summary(value=[tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])
            self.writer.add_summary(summary_test, epoch)

            line = "Epoch: [%2d], test_acc: %.4f \n" % (epoch, test_acc)
            print(line)
            with open('logs.txt', 'a') as f:
                f.write(line)

            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter)

            # show temporal results
            self.visualize_results(epoch)

            # save model for final step
        self.save(self.checkpoint_dir, counter)

    def dataset_move(self, idx):
        self.unlabelled_X = list(self.unlabelled_X)
        self.unlabelled_X = self.unlabelled_X[(idx + 1) * self.unlabelled_batch_size:] \
                            + self.unlabelled_X[
                              idx * self.unlabelled_batch_size: (idx + 1) * self.unlabelled_batch_size]
        self.unlabelled_X = np.asarray(self.unlabelled_X)

    def visualize_results(self, epoch):
        # tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(self.visual_num)))
        z_sample = np.random.uniform(-1, 1, size=(self.visual_num, self.z_dim))

        """ random noise, random discrete code, fixed continuous code """
        y = np.random.choice(self.len_discrete_code, self.visual_num)
        # 10개의 label을 batch_size만큼 생성함
        y_one_hot = np.zeros((self.visual_num, self.y_dim))
        y_one_hot[np.arange(self.visual_num), y] = 1

        samples = self.sess.run(self.fake_images, feed_dict={self.visual_z: z_sample, self.visual_y: y_one_hot})

        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                    check_folder(
                        self.result_dir + '/' + self.model_dir + '/all_classes') + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes.png')

        """ specified condition, random noise """
        n_styles = 10  # must be less than or equal to self.batch_size

        np.random.seed()
        si = np.random.choice(self.visual_num, n_styles)

        for l in range(self.len_discrete_code):
            y = np.zeros(self.visual_num, dtype=np.int64) + l
            y_one_hot = np.zeros((self.visual_num, self.y_dim))
            y_one_hot[np.arange(self.visual_num), y] = 1

            samples = self.sess.run(self.fake_images, feed_dict={self.visual_z: z_sample, self.visual_y: y_one_hot})
            save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                        check_folder(
                            self.result_dir + '/' + self.model_dir + '/class_%d' % l) + '/' + self.model_name + '_epoch%03d' % epoch + '_test_class_%d.png' % l)

            samples = samples[si, :, :, :]

            if l == 0:
                all_samples = samples
            else:
                all_samples = np.concatenate((all_samples, samples), axis=0)

        """ save merged images to check style-consistency """
        canvas = np.zeros_like(all_samples)
        for s in range(n_styles):
            for c in range(self.len_discrete_code):
                canvas[s * self.len_discrete_code + c, :, :, :] = all_samples[c * n_styles + s, :, :, :]

        save_images(canvas, [n_styles, self.len_discrete_code],
                    check_folder(
                        self.result_dir + '/' + self.model_dir + '/all_classes_style_by_style') + '/' + self.model_name + '_epoch%03d' % epoch + '_test_all_classes_style_by_style.png')

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
