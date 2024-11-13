import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from time import time

# 模型训练函数
def model_train(model, train_generator, test_generator, epochs=15):
    lr = 0.001
    opt = Adam(learning_rate=lr, decay=lr / epochs)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # 模型训练
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=test_generator,
        verbose=1
    )
    
    # 评估测试集
    score, accuracy = model.evaluate(test_generator, verbose=1)
    print("Test score is {}".format(score))
    print("Test accuracy is {}".format(accuracy))

    # 保存模型
    t = time()
    export_path = "saved_models/model_{}.h5".format(int(t))
    tf.keras.models.save_model(model, export_path)

    return history, export_path

# 绘制训练曲线
def plot_training_curves(history, epochs=15):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    # 绘制损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    # 添加自定义标题
    plt.suptitle('何昂202210310219')

    plt.show()

if __name__ == '__main__':
    # 假设你已经构建了模型 `model` 和数据生成器 `train_generator`, `test_generator`
    model = buildmodel()  # 构建模型
    history, export_path = model_train(model, train_generator, test_generator)  # 训练模型

    # 绘制训练曲线
    plot_training_curves(history)
