import os

from .srez_model import Model, _discriminator_model
import numpy as np
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

def create_wgan_model(sess, gene_input_pl, real_images_pl, channels = 3):
    """
     Create gene and disc networks
    
    Parameters:
    ----------
     gene_input_pl : tf.placeholder
         Input of generator, shape : (16, 16, 16, 3)
     real_images_pl : tf.placeholder
         Input of discriminator, shape : (16, 64, 64, 3)
    Returns:
    ----------
     list 
    """
    gene_output, gene_var_list = generator_subpixel_network(gene_input_pl, channels)

     # Discriminator with real data
    disc_real_input = tf.identity(real_images_pl, name='disc_real_input')
    # TBD: Is there a better way to instance the discriminator?
    with tf.variable_scope('disc') as scope:
        disc_real_output, disc_var_list = \
                _discriminator_model(sess, gene_input_pl, disc_real_input)

        scope.reuse_variables()
            
        disc_fake_output, _ = _discriminator_model(sess, gene_input_pl, gene_output)

    return [gene_output, gene_var_list,
            disc_real_output, disc_fake_output, disc_var_list]
def _downscale(images, K):
    """Differentiable image downscaling by a factor of K"""
    arr = np.zeros([K, K, 3, 3])
    arr[:,:,0,0] = 1.0/(K*K)
    arr[:,:,1,1] = 1.0/(K*K)
    arr[:,:,2,2] = 1.0/(K*K)
    dowscale_weight = tf.constant(arr, dtype=tf.float32)
    
    downscaled = tf.nn.conv2d(images, dowscale_weight,
                              strides=[1, K, K, 1],
                              padding='SAME')
    return downscaled

def generator_subpixel_network(features, channels):
    # Upside-down all-convolutional resnet

    mapsize = 3
    res_units  = [64, 32, 24]

    old_vars = tf.global_variables()

    # See Arxiv 1603.05027
    model = Model('GEN_subpixel', features)
    with tf.variable_scope('GEN_subpixel'):
        for ru in range(len(res_units)-1):
            nunits  = res_units[ru]
    
            for j in range(2):
                model.add_residual_block(nunits, mapsize=mapsize)
    
            # Spatial upscale (see http://distill.pub/2016/deconv-checkerboard/)
            # and transposed convolution
            # model.add_upscale()
            #import ipdb; ipdb.set_trace()
            # Spatial upscale : subpixel
            model.add_conv2d(nunits*4, mapsize=mapsize, stride=1, stddev_factor=2.)
            model.add_upscale_subpixel(nunits,r = 2, color=False)
            
            model.add_batch_norm()
            model.add_relu()
            model.add_conv2d_transpose(nunits, mapsize=mapsize, stride=1, stddev_factor=1.)
    
        # Finalization a la "all convolutional net"
        model.add_conv2d(nunits, mapsize=mapsize, stride=1, stddev_factor=2.)
        # Worse: model.add_batch_norm()
        model.add_relu()
        model.add_conv2d(nunits, mapsize=mapsize, stride=1, stddev_factor=2.)
        # Worse: model.add_batch_norm()
        model.add_relu()
        # model.add_conv2d(channels, mapsize=1, stride=1, stddev_factor=1.)
        # # Worse: model.add_batch_norm()
        # model.add_relu()
    
        # Last layer is sigmoid with no batch normalization
        # upscaling ratio
        # Last layer is sigmoid with no batch normalization
        model.add_conv2d(channels, mapsize=1, stride=1, stddev_factor=1.)

        model.add_sigmoid()
        
        new_vars  = tf.global_variables()
        gene_vars = list(set(new_vars) - set(old_vars))

    return model.get_output(), gene_vars
def generator_subpixel_loss_wgan(disc_output, gene_output, features,labels):
    # I.e. did we fool the discriminator?
    # cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits = disc_output,
    #                                                         labels = tf.ones_like(disc_output))
    # gene_ce_loss  = tf.reduce_mean(cross_entropy, name='gene_ce_loss')
    # Generator loss for wgan
    gene_wgan_loss = - tf.reduce_mean(disc_output)
    # I.e. does the result look like the feature?
    K = int(gene_output.get_shape()[1])//int(features.get_shape()[1])
    assert K == 2 or K == 4 or K == 8    
    downscaled = _downscale(gene_output, K)

    # cosine distance
    batch_size, h, w, channel = gene_output.shape.as_list()
    gene_output_unit = tf.reshape(gene_output, [batch_size, -1])
    gene_output_unit = tf.contrib.layers.unit_norm(inputs = gene_output_unit, dim = 1)
    labels_unit = tf.reshape(labels, [batch_size, -1])
    labels_unit = tf.contrib.layers.unit_norm(inputs = labels_unit, dim = 1)
    gene_cos_loss = tf.losses.cosine_distance(labels = labels_unit,
                                              predictions = gene_output_unit, dim = 1)
    
    
    # subtract real image
    # gene_l1_loss  = tf.reduce_mean(tf.abs(gene_output - labels), name='gene_l1_loss')
    # gene_loss     = tf.add((1.0 - FLAGS.gene_l1_factor) * gene_wgan_loss,
    #                        FLAGS.gene_l1_factor * gene_l1_loss, name='gene_loss')

    gene_loss     = tf.add((1.0 - FLAGS.gene_l1_factor) * gene_wgan_loss,
                           FLAGS.gene_l1_factor * gene_cos_loss, name='gene_loss')

    return gene_loss

def gan_generator_subpixel_loss(gene_output, real_images_pl,
                                disc_output, gene_l1_factor):
    # I.e. did we fool the discriminator?
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits = disc_output,
                                                            labels = tf.ones_like(disc_output))
    gene_ce_loss  = tf.reduce_mean(cross_entropy, name='gene_ce_loss')

    # I.e. does the result look like the feature?
    # K = int(gene_output.get_shape()[1])//int(train_features_pl.get_shape()[1])
    # assert K == 2 or K == 4 or K == 8    
    # downscaled = _downscale(gene_output, K)
    # gene_l1_loss  = tf.reduce_mean(tf.abs(downscaled - train_features_pl), name='gene_l1_loss')    
    # gene_l2_loss  = tf.reduce_mean(tf.square(gene_output - real_images_pl), name='gene_l2_loss')
    gene_l1_loss  = tf.reduce_mean(tf.abs(gene_output - real_images_pl), name='gene_l1_loss')

    gene_loss     = tf.add((1.0 - gene_l1_factor) * gene_ce_loss,
                           gene_l1_factor * gene_l1_loss, name='gene_loss')

    return gene_loss
def discriminator_loss_wgan(disc_real_output, disc_fake_output):
    # I.e. did we correctly identify the input as real or not?
    # cross_entropy_real = tf.nn.sigmoid_cross_entropy_with_logits(logits = disc_real_output,
    #                                                              labels = tf.ones_like(disc_real_output))
    # disc_real_loss     = tf.reduce_mean(cross_entropy_real, name='disc_real_loss')
    
    # cross_entropy_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits = disc_fake_output,
    #                                                              labels = tf.zeros_like(disc_fake_output))
    # disc_fake_loss     = tf.reduce_mean(cross_entropy_fake, name='disc_fake_loss')
    # Discriminator loss for wgan
    disc_real_loss = - tf.reduce_mean(disc_real_output)
    disc_fake_loss = tf.reduce_mean(disc_fake_output)
    return disc_real_loss, disc_fake_loss

def gan_discriminator_loss(disc_real_output, disc_fake_output):
    # I.e. did we correctly identify the input as real or not?
    cross_entropy_real = tf.nn.sigmoid_cross_entropy_with_logits(logits = disc_real_output,
                                                                 labels = tf.ones_like(disc_real_output))
    disc_real_loss     = tf.reduce_mean(cross_entropy_real, name='disc_real_loss')
    
    cross_entropy_fake = tf.nn.sigmoid_cross_entropy_with_logits(logits = disc_fake_output,
                                                                 labels = tf.zeros_like(disc_fake_output))
    disc_fake_loss     = tf.reduce_mean(cross_entropy_fake, name='disc_fake_loss')

    # Discriminator loss for wgan
    # disc_real_loss = - tf.reduce_mean(disc_real_output)
    # disc_fake_loss = tf.reduce_mean(disc_fake_output)
    return disc_real_loss, disc_fake_loss

def gan_adam_optimizers(gene_loss, gene_var_list,
                        disc_loss, disc_var_list,
                        learning_rate, learning_beta1):    
    # TBD: Does this global step variable need to be manually incremented? I think so.
    # global_step    = tf.Variable(0, dtype=tf.int64,   trainable=False, name='global_step')
     
    gene_opti = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=learning_beta1,
                                       name='gene_optimizer')
    disc_opti = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=learning_beta1,
                                       name='disc_optimizer')

    gene_minimize = gene_opti.minimize(gene_loss, var_list = gene_var_list, \
                                       name = 'gene_loss_minimize')
    
    disc_minimize = disc_opti.minimize(disc_loss, var_list=disc_var_list, \
                                           name = 'disc_loss_minimize')#, global_step = global_step)
    
    return (gene_minimize, disc_minimize)

def create_optimizers_wgan(gene_loss, gene_var_list,
                           disc_loss, disc_var_list,
                           learning_rate):    
    # TBD: Does this global step variable need to be manually incremented? I think so.
    # global_step    = tf.Variable(0, dtype=tf.int64,   trainable=False, name='global_step')
     
    # gene_opti = tf.train.AdamOptimizer(learning_rate=learning_rate_pl,
    #                                    beta1=FLAGS.learning_beta1,
    #                                    name='gene_optimizer')
    # disc_opti = tf.train.AdamOptimizer(learning_rate=learning_rate_pl,
    #                                    beta1=FLAGS.learning_beta1,
    #                                    name='disc_optimizer')
    gene_opti = tf.train.RMSPropOptimizer(learning_rate = learning_rate, \
                                          name = 'gene_optimizer_RMS')
    disc_opti = tf.train.RMSPropOptimizer(learning_rate = learning_rate, \
                                          name = 'disc_optimizer_RMS')

    gene_minimize = gene_opti.minimize(gene_loss, var_list = gene_var_list, \
                                       name = 'gene_loss_minimize')
    
    disc_minimize     = disc_opti.minimize(disc_loss, var_list=disc_var_list, \
                                           name = 'disc_loss_minimize')#, global_step = global_step)
    
    return (gene_minimize, disc_minimize)
