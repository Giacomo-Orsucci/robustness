import matplotlib.pyplot as plt


def plotting(std_array, accuracy_array, psnr, x1_title, y1_title, y2_title, image_title):


    #std_array = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    #accuracy_array = [0.9972871895468235, 0.9614369208879388, 0.8513202080844546, 0.7599712079560951, 0.6991411828876849, 0.657818567729759, 0.628944142462253, 0.6068721658029914, 0.5907433247767806, 0.5773794022230629, 0.5667455349689067]
    #psnr = [0,10,20,30,40,50,60,70,80,90,100]


    #plt.plot(std_array, accuracy_array, marker='s', linestyle='--', color='black', markerfacecolor='red', markeredgecolor='red')
    #plt.grid(color='grey', linestyle='-', linewidth=0.5)



    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax2 = ax1.twinx()


    PSNR, = ax2.plot(std_array, psnr, marker='s', linestyle='--', color='black', markerfacecolor='blue', markeredgecolor='blue', label='PSNR')
    Accuracy, = ax1.plot(std_array, accuracy_array, marker='s', linestyle='--', color='black', markerfacecolor='red', markeredgecolor='red', label='Accuracy')

    ax1.set_xlabel(x1_title)
    ax1.set_ylabel(y1_title)
    ax1.tick_params(axis="y")
    ax2.set_ylabel(y2_title)
    ax1.set_yticks([0.4,0.5,0.6,0.7,0.8,0.9,1.0]) #to fix the y scale but it can be used also accuracy_array


    #fig.legend(handles=[PSNR, Accuracy])

    # Create legend for both plots
    ax1.legend(handles=[PSNR, Accuracy], shadow=True, bbox_to_anchor=(0.05, -0.15), loc="lower right")
    ax1.grid(color='grey', linestyle='-', linewidth=0.5)


    plt.title(image_title, fontweight="bold")
    #fig.autofmt_xdate()
    plt.gca().invert_xaxis() #only for jpeg_compression
    plt.show()

def main():

    std_array = [128, 120, 112, 104, 96, 88, 80, 72, 64, 56, 48, 40, 32, 24, 16]
    accuracy_array = [0.9972875895476341, 0.9956649306300925, 0.9883758872778472, 0.9720545378908515, 0.928416025330956, 0.8920603498067794, 0.8263934205045387, 0.7971781036972008, 0.7644057249799207, 0.7329120722937102, 0.7413474466707061, 0.6423652110996769, 0.5645748364399642, 0.5672296777057398, 0.5091053290472323]
    psnr_array = [100.0020000400008, 36.8871409009813, 34.011798825940794, 32.39019663707556, 31.286375546408888, 30.469551089651905, 29.836456437258736, 29.33560560763911, 28.939649703288076, 28.621827091594508, 28.36398960260379, 28.155095063503854, 27.989681997596115, 27.86552969912534, 27.779711014766036]

    plotting(std_array, accuracy_array, psnr_array, "Crop size","Bitwise accuracy","PSNR (dB)","Center cropping")


if __name__ == "__main__": 
    main() 