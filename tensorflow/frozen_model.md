### tf1.x查看pb文件所有input node name
```
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
