### tf1.x查看pb文件所有input node name
```python
import tensorflow as tf

def inspect_pb_inputs(pb_file_path):
    """
    解析TensorFlow 1.x的pb文件，查看所有输入节点及其名称
    
    Args:
        pb_file_path: pb模型文件的路径
    """
    # 重置默认图
    tf.reset_default_graph()
    
    # 读取pb文件
    with tf.gfile.GFile(pb_file_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    
    # 将graph_def导入当前图
    tf.import_graph_def(graph_def, name="")
    
    # 获取当前图的所有节点
    graph = tf.get_default_graph()
    input_nodes = []
    
    # 遍历所有节点，筛选输入节点（Placeholder类型）
    for node in graph_def.node:
        # Placeholder是TF1.x中最常见的输入节点类型
        if node.op == "Placeholder":
            input_nodes.append({
                "name": node.name,          # 节点名称（部署时常用）
                "op_type": node.op,         # 节点操作类型
                "dtype": node.attr["dtype"].type,  # 输入数据类型
                "shape": node.attr["shape"].shape  # 输入形状（可选）
            })
    
    # 打印输入节点信息
    print("=== PB文件输入节点信息 ===")
    if not input_nodes:
        print("未找到Placeholder类型的输入节点（可能是其他类型输入）")
    else:
        for idx, node in enumerate(input_nodes):
            print(f"\n输入节点 {idx+1}:")
            print(f"  节点名称 (name): {node['name']}")
            print(f"  操作类型: {node['op_type']}")

            
inspect_pb_inputs('prm_model.pb')
```



### PRM模型冻结为pb文件并用来推理的例子
```python
import itertools
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, static_bidirectional_rnn, LSTMCell, MultiRNNCell
import numpy as np
import heapq

# 原有工具函数保持不变
def tau_function(x):
    return tf.where(x > 0, tf.exp(x), tf.zeros_like(x))

def attention_score(x):
    return tau_function(x) / tf.add(tf.reduce_sum(tau_function(x), axis=1, keepdims=True), 1e-20)

class BaseModel(object):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, itm_spar_num, itm_dens_num,
                 profile_num, max_norm=None):
        # reset graph
        tf.reset_default_graph()
        self.graph = tf.Graph()
        with self.graph.as_default():

            # input placeholders
            with tf.name_scope('inputs'):
                self.itm_spar_ph = tf.placeholder(tf.int32, [None, max_time_len, itm_spar_num], name='item_spar')
                self.itm_dens_ph = tf.placeholder(tf.float32, [None, max_time_len, itm_dens_num], name='item_dens')
                self.usr_profile = tf.placeholder(tf.int32, [None, profile_num], name='usr_profile')
                self.seq_length_ph = tf.placeholder(tf.int32, [None, ], name='seq_length_ph')
                self.label_ph = tf.placeholder(tf.float32, [None, max_time_len], name='label_ph')

                self.is_train = tf.placeholder_with_default(False, [], name='is_train')
                self.lr = tf.placeholder_with_default(0.0, [], name='lr')
                self.reg_lambda = tf.placeholder_with_default(0.0, [], name='reg_lambda')
                self.keep_prob = tf.placeholder_with_default(1.0, [], name='keep_prob')
                self.max_time_len = max_time_len
                self.hidden_size = hidden_size
                self.emb_dim = eb_dim
                self.itm_spar_num = itm_spar_num
                self.itm_dens_num = itm_dens_num
                self.profile_num = profile_num
                self.max_grad_norm = max_norm
                self.ft_num = itm_spar_num * eb_dim + itm_dens_num
                self.feature_size = feature_size

            # embedding
            with tf.name_scope('embedding'):
                self.emb_mtx = tf.get_variable('emb_mtx', [feature_size + 1, eb_dim],
                                               initializer=tf.truncated_normal_initializer)
                self.itm_spar_emb = tf.gather(self.emb_mtx, self.itm_spar_ph)
                self.usr_prof_emb = tf.gather(self.emb_mtx, self.usr_profile)

                # 将用户画像嵌入展平并复制到每个时间步
                usr_prof_flat = tf.reshape(self.usr_prof_emb, [-1, self.profile_num * self.emb_dim])
                usr_prof_tiled = tf.tile(tf.expand_dims(usr_prof_flat, axis=1), [1, self.max_time_len, 1])
                self.item_seq = tf.concat(
                    [tf.reshape(self.itm_spar_emb, [-1, self.max_time_len, self.itm_spar_num * self.emb_dim]),
                     self.itm_dens_ph,
                     usr_prof_tiled], axis=-1)

    # ---------- 以下为原有网络构建方法，未作修改 ----------
    def build_fc_net(self, inp, scope='fc'):
        with tf.variable_scope(scope):
            bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1', training=self.is_train)
            fc1 = tf.layers.dense(bn1, 200, activation=tf.nn.relu, name='fc1')
            dp1 = tf.nn.dropout(fc1, self.keep_prob, name='dp1')
            fc2 = tf.layers.dense(dp1, 80, activation=tf.nn.relu, name='fc2')
            dp2 = tf.nn.dropout(fc2, self.keep_prob, name='dp2')
            fc3 = tf.layers.dense(dp2, 2, activation=None, name='fc3')
            score = tf.nn.softmax(fc3)
            score = tf.reshape(score[:, :, 0], [-1, self.max_time_len])
            seq_mask = tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)
            y_pred = seq_mask * score
        return y_pred

    def build_mlp_net(self, inp, layer=(500, 200, 80), scope='mlp'):
        with tf.variable_scope(scope):
            inp = tf.layers.batch_normalization(inputs=inp, name='mlp_bn', training=self.is_train)
            for i, hidden_num in enumerate(layer):
                fc = tf.layers.dense(inp, hidden_num, activation=tf.nn.relu, name='fc' + str(i))
                inp = tf.nn.dropout(fc, self.keep_prob, name='dp' + str(i))
            final = tf.layers.dense(inp, 1, activation=None, name='fc_final')
            score = tf.reshape(final, [-1, self.max_time_len])
            seq_mask = tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)
            y_pred = seq_mask * score
        return y_pred

    def build_logloss(self, y_pred):
        self.loss = tf.losses.log_loss(self.label_ph, y_pred)
        self.opt()

    def build_norm_logloss(self, y_pred):
        self.loss = - tf.reduce_sum(self.label_ph/(tf.reduce_sum(self.label_ph, axis=-1, keepdims=True) + 1e-8) * tf.log(y_pred))
        self.opt()

    def build_mseloss(self, y_pred):
        self.loss = tf.losses.mean_squared_error(self.label_ph, y_pred)
        self.opt()

    def build_attention_loss(self, y_pred):
        self.label_wt = attention_score(self.label_ph)
        self.pred_wt = attention_score(y_pred)
        self.loss = tf.losses.log_loss(self.label_wt, self.pred_wt)
        self.opt()

    def opt(self):
        for v in tf.trainable_variables():
            if 'bias' not in v.name and 'emb' not in v.name:
                self.loss += self.reg_lambda * tf.nn.l2_loss(v)

        self.optimizer = tf.train.AdamOptimizer(self.lr)

        if self.max_grad_norm > 0:
            grads_and_vars = self.optimizer.compute_gradients(self.loss)
            for idx, (grad, var) in enumerate(grads_and_vars):
                if grad is not None:
                    grads_and_vars[idx] = (tf.clip_by_norm(grad, self.max_grad_norm), var)
            self.train_step = self.optimizer.apply_gradients(grads_and_vars)
        else:
            self.train_step = self.optimizer.minimize(self.loss)

    def multihead_attention(self,
                            queries,
                            keys,
                            num_units=None,
                            num_heads=2,
                            scope="multihead_attention",
                            reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            if num_units is None:
                num_units = queries.get_shape().as_list()[-1]

            Q = tf.layers.dense(queries, num_units, activation=None)
            K = tf.layers.dense(keys, num_units, activation=None)
            V = tf.layers.dense(keys, num_units, activation=None)

            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)

            outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
            outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

            key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))
            key_masks = tf.tile(key_masks, [num_heads, 1])
            key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])

            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)
            outputs = tf.nn.softmax(outputs)

            query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))
            query_masks = tf.tile(query_masks, [num_heads, 1])
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
            outputs *= query_masks
            outputs = tf.nn.dropout(outputs, self.keep_prob)
            outputs = tf.matmul(outputs, V_)
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)

        return outputs

    def positionwise_feed_forward(self, inp, d_hid, d_inner_hid, dropout=0.9):
        with tf.variable_scope('pos_ff'):
            inp = tf.layers.batch_normalization(inputs=inp, name='bn1', training=self.is_train)
            l1 = tf.layers.conv1d(inp, d_inner_hid, 1, activation='relu')
            l2 = tf.layers.conv1d(l1, d_hid, 1)
            dp = tf.nn.dropout(l2, dropout, name='dp')
            dp = dp + inp
            output = tf.layers.batch_normalization(inputs=dp, name='bn2', training=self.is_train)
        return output

    def bilstm(self, inp, hidden_size, scope='bilstm', reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, name='cell_fw')
            lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size, forget_bias=1.0, name='cell_bw')
            outputs, state_fw, state_bw = static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, inp, dtype='float32')
        return outputs, state_fw, state_bw

    def train(self, batch_data, lr, reg_lambda, keep_prob=0.8):
        with self.graph.as_default():
            loss, _ = self.sess.run([self.loss, self.train_step], feed_dict={
                self.usr_profile: np.reshape(np.array(batch_data[1]), [-1, self.profile_num]),
                self.itm_spar_ph: batch_data[2],
                self.itm_dens_ph: batch_data[3],
                self.label_ph: batch_data[4],
                self.seq_length_ph: batch_data[6],
                self.lr: lr,
                self.reg_lambda: reg_lambda,
                self.keep_prob: keep_prob,
                self.is_train: True,
            })
            return loss

    def eval(self, batch_data, reg_lambda, keep_prob=1, no_print=True):
        with self.graph.as_default():
            pred, loss = self.sess.run([self.y_pred, self.loss], feed_dict={
                self.usr_profile: np.reshape(np.array(batch_data[1]), [-1, self.profile_num]),
                self.itm_spar_ph: batch_data[2],
                self.itm_dens_ph: batch_data[3],
                self.label_ph: batch_data[4],
                self.seq_length_ph: batch_data[6],
                self.reg_lambda: reg_lambda,
                self.keep_prob: keep_prob,
                self.is_train: False,
            })
            return pred.reshape([-1, self.max_time_len]).tolist(), loss

    def save(self, path):
        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.save(self.sess, save_path=path)
            print('Save model:', path)

    def load(self, path):
        with self.graph.as_default():
            ckpt = tf.train.get_checkpoint_state(path)
            if ckpt and ckpt.model_checkpoint_path:
                saver = tf.train.Saver()
                saver.restore(sess=self.sess, save_path=ckpt.model_checkpoint_path)
                print('Restore model:', ckpt.model_checkpoint_path)

    def set_sess(self, sess):
        self.sess = sess


class PRM(BaseModel):
    def __init__(self, feature_size, eb_dim, hidden_size, max_time_len, itm_spar_num, itm_dens_num,
                 profile_num, max_norm=None, d_model=64, d_inner_hid=128, n_head=1):
        super(PRM, self).__init__(feature_size, eb_dim, hidden_size, max_time_len,
                                 itm_spar_num, itm_dens_num, profile_num, max_norm)

        with self.graph.as_default():
            pos_dim = self.item_seq.get_shape().as_list()[-1]
            self.d_model = d_model
            self.pos_mtx = tf.get_variable("pos_mtx", [max_time_len, pos_dim],
                                       initializer=tf.truncated_normal_initializer)
            self.item_seq = self.item_seq + self.pos_mtx
            if pos_dim % 2:
                self.item_seq = tf.pad(self.item_seq, [[0, 0], [0, 0], [0, 1]])

            self.item_seq = self.multihead_attention(self.item_seq, self.item_seq, num_units=d_model, num_heads=n_head)
            self.item_seq = self.positionwise_feed_forward(self.item_seq, self.d_model, d_inner_hid, self.keep_prob)

            mask = tf.expand_dims(tf.sequence_mask(self.seq_length_ph, maxlen=max_time_len, dtype=tf.float32), axis=-1)
            seq_rep = self.item_seq * mask

            # 修改点：给输出张量显式命名，便于导出 PB 时定位
            self.y_pred = tf.identity(self.build_prm_fc_function(seq_rep), name='output')
            self.build_logloss(self.y_pred)

            # 创建会话并初始化变量
            self.sess = tf.Session(graph=self.graph)
            self.sess.run(tf.global_variables_initializer())

    def build_prm_fc_function(self, inp):
        bn1 = tf.layers.batch_normalization(inputs=inp, name='bn1', training=self.is_train)
        fc1 = tf.layers.dense(bn1, self.d_model, activation=tf.nn.relu, name='fc1')
        dp1 = tf.nn.dropout(fc1, self.keep_prob, name='dp1')
        fc2 = tf.layers.dense(dp1, 1, activation=None, name='fc2')
        score = tf.nn.softmax(tf.reshape(fc2, [-1, self.max_time_len]))
        seq_mask = tf.sequence_mask(self.seq_length_ph, maxlen=self.max_time_len, dtype=tf.float32)
        return seq_mask * score

    # ---------- 新增方法：将当前模型保存为 PB 文件 ----------
    def save_as_pb(self, pb_path, output_node_names=['output']):
        """
        将当前会话中的模型冻结并保存为 PB 文件
        :param pb_path: 保存路径，例如 './model.pb'
        :param output_node_names: 输出节点名称列表，默认为 ['output']
        """
        with self.graph.as_default():
            # 将图中的所有变量转换为常量
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                self.sess,
                self.graph.as_graph_def(),
                output_node_names
            )
            # 写入文件
            with tf.gfile.GFile(pb_path, 'wb') as f:
                f.write(output_graph_def.SerializeToString())
            print('模型已保存为 PB 文件：{}'.format(pb_path))

    # ---------- 可选：加载 PB 并进行预测的静态方法 ----------
    @staticmethod
    def predict_with_pb(pb_path, input_dict, output_tensor_name='output:0'):
        """
        加载 PB 文件并执行推理
        :param pb_path: PB 文件路径
        :param input_dict: 字典，键为 placeholder 名称（字符串），值为 numpy 数组
        :param output_tensor_name: 输出张量名称，默认为 'output:0'
        :return: 预测结果 numpy 数组
        """
        with tf.gfile.GFile(pb_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
            # 获取输入输出张量
            input_tensors = {}
            for input_name in input_dict.keys():
                # 注意：导入后节点名称可能与原 placeholder 名称相同，也可能带有前缀
                # 通常原 placeholder 的名称就是 'input_name:0'，但此处我们直接通过名字获取
                tensor = graph.get_tensor_by_name(input_name + ':0')
                input_tensors[input_name] = tensor
            output_tensor = graph.get_tensor_by_name(output_tensor_name)

            with tf.Session(graph=graph) as sess:
                feed_dict = {input_tensors[name]: value for name, value in input_dict.items()}
                result = sess.run(output_tensor, feed_dict=feed_dict)
        return result


# ---------- 测试代码（放在 main 中） ----------
if __name__ == '__main__':
    # 设定模型参数（可根据需要调整）
    feature_size = 1000      # 特征总数
    eb_dim = 8               # Embedding 维度
    hidden_size = 32         # 隐藏层大小（本例中未直接使用）
    max_time_len = 10        # 序列最大长度
    itm_spar_num = 3         # 每个物品的稀疏特征个数
    itm_dens_num = 2         # 每个物品的稠密特征个数
    profile_num = 5          # 用户画像特征个数
    max_norm = 5.0           # 梯度裁剪阈值

    # 创建 PRM 模型实例（自动创建会话并初始化变量）
    model = PRM(feature_size, eb_dim, hidden_size, max_time_len,
                itm_spar_num, itm_dens_num, profile_num, max_norm,
                d_model=64, d_inner_hid=128, n_head=2)

    # 生成随机输入数据（batch_size=4）
    batch_size = 4
    fake_itm_spar = np.random.randint(0, feature_size, size=(batch_size, max_time_len, itm_spar_num))
    fake_itm_dens = np.random.randn(batch_size, max_time_len, itm_dens_num).astype(np.float32)
    fake_usr_profile = np.random.randint(0, feature_size, size=(batch_size, profile_num))
    fake_seq_len = np.random.randint(1, max_time_len+1, size=(batch_size,))

    # 使用原始模型进行一次推理（得到预测值作为基准）
    feed_dict_original = {
        model.itm_spar_ph: fake_itm_spar,
        model.itm_dens_ph: fake_itm_dens,
        model.usr_profile: fake_usr_profile,
        model.seq_length_ph: fake_seq_len,
    }
    with model.graph.as_default():
        original_output = model.sess.run(model.y_pred, feed_dict=feed_dict_original)
    print('原始模型输出形状：', original_output.shape)
    print('原始模型输出样例：\n', original_output)

    # 保存 PB 文件
    pb_save_path = './prm_model.pb'
    model.save_as_pb(pb_save_path, output_node_names=['output'])

    # 使用 PB 文件进行推理（只提供必需的四个输入）
    input_for_pb = {
        'inputs/item_spar': fake_itm_spar,      # 对应 placeholder name 'item_spar'
        'inputs/item_dens': fake_itm_dens,      # 对应 placeholder name 'item_dens'
        'inputs/usr_profile': fake_usr_profile, # 对应 placeholder name 'usr_profile'
        'inputs/seq_length_ph': fake_seq_len    # 对应 placeholder name 'seq_length_ph'
    }
    pb_output = PRM.predict_with_pb(pb_save_path, input_for_pb, output_tensor_name='output:0')
    print('PB 模型输出形状：', pb_output.shape)
    print('PB 模型输出样例：\n', pb_output)

    # 验证两个输出是否一致（允许极小误差）
    if np.allclose(original_output, pb_output, rtol=1e-5, atol=1e-5):
        print('✅ 原始模型与 PB 模型输出一致，测试通过！')
    else:
        print('❌ 输出不一致，请检查！')
        diff = np.abs(original_output - pb_output)
        print('最大差异：', np.max(diff))
```
