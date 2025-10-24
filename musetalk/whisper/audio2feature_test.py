import os
from whisper import load_model
import soundfile as sf
import numpy as np
import time
import sys
sys.path.append("..")

class Audio2Feature():
    def __init__(self, 
                 whisper_model_type="tiny",
                 model_path="./models/whisper/tiny.pt"):
        self.whisper_model_type = whisper_model_type
        self.model = load_model(model_path) #

    def get_sliced_feature(self,
                           feature_array, 
                           vid_idx, 
                           audio_feat_length=[2,2],
                           fps=25):
        """
        Get sliced features based on a given index
        :param feature_array: 
        :param start_idx: the start index of the feature
        :param audio_feat_length:
        :return: 
        """
        length = len(feature_array)  # 获取特征数组的长度
        selected_feature = []        # 用于存储选中的特征
        selected_idx = []            # 用于存储选中特征的索引
        
        center_idx = int(vid_idx * 50 / fps)  # 计算中心索引（将视频帧索引映射到50fps的音频特征轨道上）
        left_idx = center_idx - audio_feat_length[0] * 2  # 计算左边界索引（根据audio_feat_length扩展窗口长度）
        right_idx = center_idx + (audio_feat_length[1] + 1) * 2  # 计算右边界索引
        
        for idx in range(left_idx, right_idx):  # 遍历从左到右的索引区间
            idx = max(0, idx)                  # 防止索引越界到小于0
            idx = min(length - 1, idx)         # 防止索引越界到大于数组长度
            x = feature_array[idx]              # 取出对应索引的特征
            selected_feature.append(x)          # 添加到特征列表
            selected_idx.append(idx)            # 记录当前索引
        
        selected_feature = np.concatenate(selected_feature, axis=0)      # 沿第一个轴拼接特征
        selected_feature = selected_feature.reshape(-1, 384)             # (N, 384) 重新调整形状为每一帧384维
        return selected_feature, selected_idx                             # 返回最终特征及其对应索引

    def get_sliced_feature_sparse(self,feature_array, vid_idx, audio_feat_length= [2,2],fps = 25):
        """
        Get sliced features based on a given index
        :param feature_array: 
        :param start_idx: the start index of the feature
        :param audio_feat_length:
        :return: 
        """
        length = len(feature_array)
        selected_feature = []
        selected_idx = []

        for dt in range(-audio_feat_length[0],audio_feat_length[1]+1):
            left_idx = int((vid_idx+dt)*50/fps)
            if left_idx<1 or left_idx>length-1:
                print('test-----,left_idx=',left_idx)
                left_idx = max(0, left_idx)
                left_idx = min(length-1, left_idx)

                x = feature_array[left_idx]
                x = x[np.newaxis,:,:]
                x = np.repeat(x, 2, axis=0)
                selected_feature.append(x)
                selected_idx.append(left_idx)
                selected_idx.append(left_idx)
            else:
                x = feature_array[left_idx-1:left_idx+1]
                selected_feature.append(x)
                selected_idx.append(left_idx-1)
                selected_idx.append(left_idx)
        selected_feature = np.concatenate(selected_feature, axis=0)
        selected_feature = selected_feature.reshape(-1, 384)# 50*384
        return selected_feature,selected_idx
    

    def feature2chunks(self, feature_array, fps, batch_size, audio_feat_length=[2,2], start=0):
        whisper_chunks = []  # 存储每一帧对应的特征
        whisper_idx_multiplier = 50. / fps  # 计算视频帧索引和音频特征（50FPS）之间的倍数关系
        i = 0  # 帧索引计数器初始化
        #print(f"video in {fps} FPS, audio idx in 50FPS")
        for _ in range(batch_size):  # 循环 batch_size 次，提取对应的特征块
       
            selected_feature, selected_idx = self.get_sliced_feature(
                feature_array=feature_array,  # 输入的特征数组
                vid_idx=i+start,              # 当前帧索引（加上 start 偏移）
                audio_feat_length=audio_feat_length,  # 使用的音频特征长度窗口
                fps=fps                       # 视频帧率
            )
            #print(f"i:{i},selected_idx {selected_idx}")
            whisper_chunks.append(selected_feature)  # 将选中的特征加入列表
            i += 1  # 帧索引自增，处理下一帧
            

        return whisper_chunks

    def audio2feat(self,audio_path):
        # 获取音频文件的特征（通过模型transcribe处理）
        result = self.model.transcribe(audio_path)  # 使用模型对输入音频进行转录，输出包括分段信息和编码器特征
        embed_list = []  # 用于存储所有分段（segment）的嵌入特征
        for emb in result['segments']:  # 遍历每一个分段
            encoder_embeddings = emb['encoder_embeddings']  # 取出当前分段的编码器嵌入特征
            encoder_embeddings = encoder_embeddings.transpose(0,2,1,3)  # 调整维度顺序，适配后续处理
            encoder_embeddings = encoder_embeddings.squeeze(0)  # 压缩掉第0维（batch维），只留特征本体
            start_idx = int(emb['start'])  # 获取当前分段的起始位置
            end_idx = int(emb['end'])  # 获取当前分段的结束位置
            emb_end_idx = int((end_idx - start_idx)/2)  # 计算当前分段需要截取的embedding长度（简单处理，假设每2对应一个特征）
            embed_list.append(encoder_embeddings[:emb_end_idx])  # 只取所需长度的编码器嵌入部分，加入列表
        concatenated_array = np.concatenate(embed_list, axis=0)  # 将所有分段的特征在第0维拼接
        return concatenated_array  # 返回拼接后的特征数组


def test_01():
    audio_processor = Audio2Feature(model_path="../../models/whisper/whisper_tiny.pt")
    audio_path = "./test.mp3"
    array = audio_processor.audio2feat(audio_path)
    print(array.shape)
    fps = 25
    whisper_idx_multiplier = 50./fps 

    i = 0
    print(f"video in {fps} FPS, audio idx in 50FPS")
    while 1:
        start_idx = int(i * whisper_idx_multiplier)
        selected_feature,selected_idx = audio_processor.get_sliced_feature(feature_array= array,vid_idx = i,audio_feat_length=[2,2],fps=fps)
        print(f"video idx {i},\t audio idx {selected_idx},\t shape {selected_feature.shape}")
        i += 1
        if start_idx>len(array):
            break


def test_silence():
# INSERT_YOUR_CODE

    import numpy as np

    # 实例化 Audio2Feature
    audio_processor = Audio2Feature(model_path="../../models/whisper/tiny.pt")

    # 生成长度为320的静音帧（采样率16kHz，20ms帧，float32, 全为0）
    silent_frame = np.zeros(320, dtype=np.float32)

    frames = [silent_frame * 52]
    # for i in range(52):
    #     frames.append(silent_frame)

    # 测试 audio2feat 的输出（由于 audio2feat 通常输入文件路径，直接传数据按照需求仿写特征处理流程）
    # 这里仅作示例，实际audio2feat实现或许不直接接受裸数据，而是文件
    # 但我们可伪造一个短音频文件，然后测试。如果audio2feat不适用则仅演示feature2chunks
    # 假设 audio2feat 返回的输出如下（生成随机tensor以模拟其结果）:
    # array = np.random.rand(10, 384).astype(np.float32)  # 模拟10帧特征，特征维384
    inputs = np.concatenate(frames)

    array = audio_processor.audio2feat(inputs)

    print("audio2feat 输出特征 shape:", array.shape)
    print("audio2feat [0] len = ", array[0].__len__())
    print("audio2feat:", array)


    fps = 25
    whisper_idx_multiplier = 50. / fps

    # 测试 feature2chunks 的输出
    whisper_chunks = audio_processor.feature2chunks(
        feature_array=array,
        fps=fps,
        batch_size=16,  # 假设batch_size为4
        start=5,
    )
    print("feature2chunks 输出分块数量：", len(whisper_chunks))
    for idx, chunk in enumerate(whisper_chunks):
        print(f"chunk {idx} shape: {chunk.shape}, content: {chunk}")


if __name__ == "__main__":
    test_silence()

