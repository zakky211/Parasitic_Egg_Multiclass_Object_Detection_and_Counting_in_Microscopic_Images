import streamlit as st
from detect import run
from PIL import Image
# from sqlalchemy.orm import sessionmaker
# from project_orm import UserInput
# from sqlalchemy import create_engine
# import streamlit as st

import cv2
import time
import psutil
import numpy as np
import argparse
import os
import platform
import sys
from pathlib import Path
from collections import Counter
import torch.backends.cudnn as cudnn

import torch


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from utils.general import set_logging
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run2(
        weights=ROOT / 'models/hasil_train/exp25/weights/best.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/dataset.yaml',
        stframe=None,
        # stbutton=None,
        kpi5_text="",
        kpi6_text="",
        imgsz=(240, 240),  # inference size (height, width)
        conf_thres=0.7,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='cpu',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=True,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if webcam:
        view_img = check_imshow()
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} ')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
            # print(f"ini s : {s}")
            # print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            # s = s.split(":")
            # d = s[2].split(" ")
            # ektraksinya = str(d[2]+ " "+d[3]+d[4]+" "+d[5])
            # print(ektraksinya)
            # print(s)
            # s = s[1,2]
            # im0=cv2.putText(im0, str("jumlahnya : "+d[2]+" "+d[3]+d[4]+" "+d[5]), (100,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,200),5,cv2.LINE_AA)
            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                # cv2.imshow(str(p), im0)
                # if cv2.waitKey(1) == ord('q'):
                #     raise StopIteration # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
        kpi5_text.write(str(psutil.virtual_memory()[2])+"%")
        kpi6_text.write(str(psutil.cpu_percent())+"%")
        stframe.image(im0, channels="BGR",use_column_width=True)
        # stbutton.button("stop")
        # Print time (inference-only)
        # LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    # t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    # LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # print(f"Results saved to { save_dir}{s}")
        # print(f"ini s nya {ektraksinya}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)
    if vid_cap:
        vid_cap.release()
    return im0

# engine = create_engine('mysql://root:@localhost:3310/polen')
# Session = sessionmaker(bind = engine)
# sess = Session()

#--------------------------------Web Page Designing------------------------------
hide_menu_style = """
    <style>
        MainMenu {visibility: hidden;}
        
        
         div[data-testid="stHorizontalBlock"]> div:nth-child(1)
        {  
            border : 2px solid #doe0db;
            border-radius:5px;
            text-align:center;
            color:black;
            background:dodgerblue;
            font-weight:bold;
            padding: 25px;
            
        }
        
        div[data-testid="stHorizontalBlock"]> div:nth-child(2)
        {   
            border : 2px solid #doe0db;
            background:dodgerblue;
            border-radius:5px;
            text-align:center;
            font-weight:bold;
            color:black;
            padding: 25px;
            
        }
    </style>
    """

main_title = """
            <div>
                <h1 style="color:white;
                text-align:center; font-size:35px;
                margin-top:-95px;">
                Polen Detection and Counting</h1>
            </div>
            """
    
sub_title = """
            <div>
                <h6 style="color:dodgerblue;
                text-align:center;
                margin-top:-40px;">
                Pusat Penelitian Kelapa Sawit </h6>
            </div>
            """
#--------------------------------------------------------------------------------


#---------------------------Main Function for Execution--------------------------
def main():
    st.set_page_config(page_title='Polen Counting', 
                       layout = 'wide', 
                       initial_sidebar_state = 'auto')
    
    st.markdown(hide_menu_style, 
                unsafe_allow_html=True)

    st.markdown(main_title,
                unsafe_allow_html=True)

    st.markdown(sub_title,
                unsafe_allow_html=True)

    inference_msg = st.empty()
    st.sidebar.title("USER Configuration")
    # activities = ["Polen", "Kecambah"]
    # choice = st.sidebar.selectbox("Please Select Activity", activities)
    
    # if choice == "Polen":
        
    input_source = 'Image'

    conf_thres = st.sidebar.text_input("Class confidence threshold", 
                                    "0.5")


    save_output_video = st.sidebar.radio("Save output polen image?",
                                        ('Yes', 'No'))

    if save_output_video == 'Yes':
        nosave = False
        display_labels=False

    else:
        nosave = True
        display_labels = True 

    # id_persil=st.sidebar.text_input("Input ID Persil: ", "12r4rhfi3r|P")
        
    weights = "5l-19122022-evaluation-ahlul1.pt"
    device="cpu"

    # ------------------------- LOCAL IMAGE ------------------------
    if input_source == "Image":
        
        video = st.sidebar.file_uploader("Select input polen image", 
                                        type=["jpg", "jpeg","png"], 
                                        accept_multiple_files=False)

        # nama_file=video.name
        if video is not None:
            our_image = Image.open(video)
            our_image.save("data/gambarnya/{}".format(video.name))
            st.image(our_image)
        if st.sidebar.button("Start Counting"):
            
            stframe = st.empty()
            
            st.markdown("""<h4 style="color:black;">
                            Memory Overall Statistics</h4>""", 
                            unsafe_allow_html=True)
            kpi5, kpi6 = st.columns(2)

            with kpi5:
                st.markdown("""<h5 style="color:black;">
                            CPU Utilization</h5>""", 
                            unsafe_allow_html=True)
                kpi5_text = st.markdown("0")
            
            with kpi6:
                st.markdown("""<h5 style="color:black;">
                            Memory Usage</h5>""", 
                            unsafe_allow_html=True)
                kpi6_text = st.markdown("0")
            
            a = run(weights=weights, 
                source="data/gambarnya/{}".format(video.name),  
                stframe=stframe, 
                kpi5_text=kpi5_text,
                kpi6_text = kpi6_text,
                conf_thres=float(conf_thres),
                device="cpu",
                    classes=[0, 1],nosave=nosave,
                    name='{}'.format(video.name)
                    )

            inference_msg.success("Inference Complete!")
            # st.button('Download')
            # st.write(stframe)
            st.image(a, channels="BGR",use_column_width=True)
            # b= b.split(",")
            # hidup = int(b[0].split(" ")[0])
            # mati = int(b[1].split(" ")[0])
            # st.write(b)
            # st.write("Hidup : "+str(hidup))
            # st.write("mati : "+str(mati))
        # hidup = st.number_input("ketik yang hidup : ", value=0)
        # mati = st.number_input("ketik yang mati : ", value=0)
        # lainnya = st.text_area("input your location")
        # submit = st.button('predictions')
        # if submit and lainnya:
        #     try:
        #         entry = UserInput(hidup = hidup, mati = mati, informasi_tambahan = lainnya)

        #         sess.add(entry)
        #         sess.commit()
        #         st.success("data added to database")  
        #     except Exception as e:
        #         st.error(f"some error occurred : {e}")

        # if st.checkbox("view database"):
        #     results = sess.query(UserInput).all()
        #     for item in results:
        #         st.subheader(item.informasi_tambahan)
        #         st.text(item.hidup)
        #         st.text(item.mati)
                    # st.text(item.no_of_rooms)
            # st.number_input("yang hidup", value=hidup)
            # st.write("dah selesai gan")
            # with open("data/gambarnya/{}".format(video.name), "rb") as file:
            #     st.download_button(
            #             label="Download image",
            #             data=file,
            #             file_name="flower.png",
            #             mime="image/png"
            #         )
        
    torch.cuda.empty_cache()
    
    # if choice == "Kecambah":
    #     weights = "best_evaluasi6.pt"
    #     device="cpu"
    #     conf_thres = st.sidebar.text_input("Class confidence threshold", 
    #                                     "0.25")

    
    #     save_output_video = st.sidebar.radio("Save output video?",
    #                                         ('Yes', 'No'))

    #     if save_output_video == 'Yes':
    #         nosave = False
    #         display_labels=False
    
    #     else:
    #         nosave = True
    #         display_labels = True
    #     activities = ["Webcam", "Video"]
    #     choice2 = st.sidebar.selectbox("Choice Webcam or Video", activities)
    #     if choice2 == "Webcam":
    #         if st.button("Start Webcam"):
    #             stframe = st.empty()
    #             stbutton = st.empty()
    #             st.markdown("""<h4 style="color:black;">
    #                             Memory Overall Statistics</h4>""", 
    #                             unsafe_allow_html=True)
    #             kpi5, kpi6 = st.columns(2)

    #             with kpi5:
    #                 st.markdown("""<h5 style="color:black;">
    #                             CPU Utilization</h5>""", 
    #                             unsafe_allow_html=True)
    #                 kpi5_text = st.markdown("0")
                
    #             with kpi6:
    #                 st.markdown("""<h5 style="color:black;">
    #                             Memory Usage</h5>""", 
    #                             unsafe_allow_html=True)
    #                 kpi6_text = st.markdown("0")
    #             v = run2(weights=weights, 
    #                 source='0',  
    #                 stframe=stframe, 
    #                 kpi5_text=kpi5_text,
    #                 # stbutton=stbutton,
    #                 kpi6_text = kpi6_text,
    #                 conf_thres=float(conf_thres),
    #                 device="cpu",
    #                 classes=15,nosave=nosave,
    #                 )
    #             st.image(v, channels="BGR",use_column_width=True)
    #             inference_msg.success("Inference Complete!")
    #         if st.button("stop"):
    #             raise StopIteration
    #     torch.cuda.empty_cache()
    # --------------------------------------------------------------



# --------------------MAIN FUNCTION CODE------------------------
if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
# ------------------------------------------------------------------
