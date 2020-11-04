#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"

#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#endif

#include <chrono>
#include <ctime>

#include <sys/stat.h>
#include <sys/types.h>

#define ENABLE_LOG 1

using namespace std;
using namespace cv;
using namespace cv::detail;

static void printUsage(char** argv)
{
    cout <<
        "Mapeado del terreno con un vídeo capturado desde un Dron.\n\n"
         << argv[0] << " --video <Nombre del Vídeo>\n\n"
        "Parámetros Opcionales:\n"
        "  --segundos <int>\n"
        "      Mapea el terreno de los primeros segundos especificados.\n"
        "  --match_conf <float>\n"
        "      Confianza para el emparejamiento y el selector de frames clave (0,1]\n"
        "      Por defecto es 0.65.\n"
        "  --compose_megapix <float>\n"
        "      Resolución para el refinador de esquinas. -1 para resolución original\n"
        "      Por defecto es -1.\n"
        "  --output <result_img>\n"
        "      Nombre del plano resultante a guardar.\n"
        "      Por defecto es 'result.jpg'.\n"
        "  --use_composer\n"
        "      Usar efinador de bordes\n"
        "      Por defecto es false\n"



        ;
}


// Argumentos (Parámetros) por defecto
vector<String> img_names;
String video_name = "";
bool try_cuda = false;
double work_megapix = 0.6;
double compose_megapix = -1;
bool use_composer = false;
float conf_thresh = 0.3; //1.f
#ifdef HAVE_OPENCV_XFEATURES2D
string features_type = "surf";
#else
string features_type = "orb";
#endif
float match_conf = 0.65f;
int range_width = -1;
string ba_cost_func = "affine";
string ba_refine_mask = "xxxxx"; //<fx><skew><ppx><aspect><ppy> x->refinar _->no refinar
string result_name = "result.jpg";

std::string folderName = "./usedFrames/";
int segundos = -1;


////FUNCTIONS

void logFile(string toSave, cv::Mat imageParams = cv::Mat()){

    std::ofstream logfich("logfile.txt", std::ios_base::app | std::ios_base::out);
    logfich << toSave;

    if(!imageParams.empty()){
        logfich << imageParams;
        logfich << "\n";
    }

    logfich.close();

}

int initialFrames(){
    system("exec rm -r ./usedFrames/*"); //BORRA todos los frames almacenados

    long int primerosiFrames = -1;


    cv::VideoCapture capture(video_name);
    if(!capture.isOpened()){
        std::cout<<"Error abriendo video...parando la ejecución"<<std::endl;
        logFile("Error abriendo video...parando la ejecución\n");
        exit(-1);
    }

    if (segundos != -1){
        primerosiFrames = (long int)capture.get(CAP_PROP_FPS) * (long int)segundos;
    }

    int num_images = 0;
    while (capture.grab()){
        cv::waitKey(capture.get(CAP_PROP_FPS));
        cv::Mat frame;
        capture.retrieve(frame);

        imwrite(folderName + std::to_string(num_images) + ".jpg", frame);

        img_names.push_back( std::to_string(num_images) );
        num_images++;

        if ( (primerosiFrames != -1) && (primerosiFrames == num_images)){
            break; //Sale del bucle for, ya ha capturado los frames que hay en x segundos del video
        }
    }
    capture.release();

    return num_images;

}

int computeMatches(cv::Mat FrameA, cv::Mat FrameB){

    Ptr<Feature2D> finder = xfeatures2d::SURF::create();
    ImageFeatures featuresFrameA, featuresFrameB;
    computeImageFeatures(finder, FrameA, featuresFrameA);
    computeImageFeatures(finder, FrameB, featuresFrameB);
    
    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( featuresFrameA.descriptors, featuresFrameB.descriptors, matches );

    //Se calculan distancias minimas y maximas entre keypoints
    double max_dist = 0; double min_dist = 100;
    for( int i = 0; i < featuresFrameA.descriptors.rows; i++ ){ 
        double dist = matches[i].distance;
        if( dist < min_dist ){ min_dist = dist; }
        if( dist > max_dist ){ max_dist = dist; }
    }

    //Se usan los buenos emparejamientos --> distancia menor a 2*min_dist
    std::vector< DMatch > good_matches;
    cv::Mat result;
    for( int i = 0; i < featuresFrameA.descriptors.rows; i++ ){ 

        if( matches[i].distance <= std::max(2*min_dist, 0.02) ){
            good_matches.push_back( matches[i]);
        }

    }
    logFile("GOOD MATCHES: " + to_string(good_matches.size()) + "\n");

    return good_matches.size();

}

std::vector<cv::Mat> keyFramesStoredSelector(std::string folderName, int & num_images){

    std::vector<cv::Mat> KeyFrames;

    cv::Mat KeyFrame = imread(folderName + img_names[0] + ".jpg");
    KeyFrames.push_back(KeyFrame);

    int MaxMatches = computeMatches(KeyFrame, imread(folderName + img_names[1] + ".jpg"));

    for(int i = 2; i < num_images; i++){

        int NewMatches = computeMatches(KeyFrame, imread(folderName + img_names[i] + ".jpg"));

        if( NewMatches < MaxMatches * match_conf){
            KeyFrames.push_back(imread(folderName + img_names[i] + ".jpg"));
            MaxMatches = computeMatches(imread(folderName + img_names[i-1] + ".jpg"), imread(folderName + img_names[i] + ".jpg")); //compute el nuevo maximo con el anterior
            KeyFrame = imread(folderName + img_names[i] + ".jpg");
        }

    }

    logFile("Nuevo numero de imágenes --> " + to_string(KeyFrames.size()) + "\n");
    num_images = KeyFrames.size();

    return KeyFrames;

}



void storeFrames(std::string folderName, std::vector<cv::Mat> frames){

    for(int i = 0 ; i < frames.size(); i++){
        imwrite(folderName + std::to_string(i) + ".jpg", frames[i]);
    }
    
    logFile("Todos los Frames se han almacenado en la carpeta --> " + folderName + "\n");

}

cv::Mat getFramei(std::string folderName, int i){

    return (imread(folderName + std::to_string(i) + ".jpg"));

}


//Utiliza las imágenes almacenadas en Carpeta
void getFeatures(int num_images, double & work_scale, double & seam_scale, bool & is_work_scale_set, bool & is_seam_scale_set, cv::Mat & full_img, cv::Mat & img, vector<ImageFeatures> & features, vector<Mat> & images, vector<Size> & full_img_sizes, double & seam_work_aspect){

    logFile("Buscando características en los frames...\n");

    double seam_megapix = 0.1;
    Ptr<Feature2D> finder;
    if (features_type == "orb"){
        finder = ORB::create();
    }
    else if (features_type == "akaze"){
        finder = AKAZE::create();
    }
    #ifdef HAVE_OPENCV_XFEATURES2D
        else if (features_type == "surf")
        {
            finder = xfeatures2d::SURF::create();
        }
        else if (features_type == "sift") {
            finder = xfeatures2d::SIFT::create();
        }
    #endif
    else
    {
        cout << "Tipo de features desconocido; '" << features_type << "'\n";
        exit(-1);
    }

    for (int i = 0; i < num_images; ++i)
    {
        full_img = getFramei(folderName, i);
        full_img_sizes[i] = full_img.size();

        if (full_img.empty())
        {
            logFile("Error abriendo imagen: " + to_string(i) + " ...saliendo\n");
            cout << "Error abriendo imagen: " << to_string(i) << " ...saliendo\n";
            exit(-1);
        }
        if (work_megapix < 0)
        {
            img = full_img;
            work_scale = 1;
            is_work_scale_set = true;
        }
        else
        {
            if (!is_work_scale_set)
            {
                work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
                is_work_scale_set = true;
            }
            resize(full_img, img, Size(), work_scale, work_scale, INTER_LINEAR_EXACT);
        }
        if (!is_seam_scale_set)
        {
            seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
            seam_work_aspect = seam_scale / work_scale;
            is_seam_scale_set = true;
        }
        computeImageFeatures(finder, img, features[i]);

        features[i].img_idx = i;
        logFile("Características en la imágen #" + to_string(i+1) + ": " + to_string(features[i].keypoints.size()) + "\n");

        resize(full_img, img, Size(), seam_scale, seam_scale, INTER_LINEAR_EXACT);
        images[i] = img.clone();
    }

}

void getMatcher(vector<MatchesInfo> & pairwise_matches, vector<ImageFeatures> & features, vector<int> & indices, vector<Mat> & images, vector<Size> & full_img_sizes){

    logFile("Emparejando pares de imágenes...\n\n");

    Ptr<FeaturesMatcher> matcher = makePtr<AffineBestOf2NearestMatcher>(false, try_cuda, match_conf);;

    (*matcher)(features, pairwise_matches);
    matcher->collectGarbage();

    // Solo se usan las imágenes que detecta el matcher
    indices = leaveBiggestComponent(features, pairwise_matches, conf_thresh);
    
    vector<Mat> img_subset;
    vector<Size> full_img_sizes_subset;
    for (size_t i = 0; i < indices.size(); ++i)
    {
        img_subset.push_back(images[indices[i]]);
        full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
    }

    images = img_subset;
    full_img_sizes = full_img_sizes_subset;  

}

void homographyEstimation(vector<ImageFeatures> & features, vector<CameraParams> & cameras, vector<MatchesInfo> & pairwise_matches, int & num_images){

    Ptr<Estimator> estimator = makePtr<AffineBasedEstimator>();
    //estimator = makePtr<HomographyBasedEstimator>();

    if (!(*estimator)(features, pairwise_matches, cameras))
    {
        cout << "Fallo en la estimación...Saliendo\n";
        logFile("Fallo en la estimación...Saliendo\n");
        exit(-1);
    }

    if(cameras.size() != num_images){
        estimator = makePtr<HomographyBasedEstimator>();
        if (!(*estimator)(features, pairwise_matches, cameras))
        {
            cout << "Fallo en la estimación...Saliendo\n";
            logFile("Fallo en la estimación...Saliendo\n");
            exit(-1);
        }
    }

    features.clear();

}

void cameraFocals(vector<CameraParams> & cameras, vector<int> & indices, vector<double> & focals){

    for (size_t i = 0; i < cameras.size(); ++i)
    {
        Mat R;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].R = R;
        logFile("Parametros intrínsicos iniciales del frame #" + to_string(indices[i]+1));
        logFile(":\nK:\n", cameras[i].K());
        logFile("\nR:\n", cameras[i].R);
    }

    // Find median focal length
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        logFile("Parametros intrínsicos del frame #" + to_string(indices[i]+1));
        logFile(":\nK:\n", cameras[i].K());
        logFile("\nR:\n", cameras[i].R);
        focals.push_back(cameras[i].focal);
    }

    sort(focals.begin(), focals.end());

}

void warpedImageScale(float & warped_image_scale, vector<double> & focals){

    if (focals.size() % 2 == 1)
        warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
    else
        warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;
   
}

void warpImages(vector<Point> & corners, vector<UMat> & masks_warped, vector<UMat> & images_warped, vector<Size> & sizes, vector<Mat> & images, int num_images, Ptr<WarperCreator> & warper_creator, Ptr<RotationWarper> & warper, float & warped_image_scale, double & seam_work_aspect, vector<CameraParams> & cameras, vector<UMat> & images_warped_f){


    logFile("Función Warp a subset de imágenes...\n");

    vector<UMat> masks(num_images);

    // Prepare images masks
    for (int i = 0; i < num_images; ++i)
    {
        masks[i].create(images[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }

    // Warp a las imágenes y sus máscaras
    warper_creator = makePtr<cv::AffineWarper>();

    warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));

    for (int i = 0; i < num_images; ++i)
    {
        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);
        float swa = (float)seam_work_aspect;
        K(0,0) *= swa; K(0,2) *= swa;
        K(1,1) *= swa; K(1,2) *= swa;
        corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
        sizes[i] = images_warped[i].size();
        warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
    }
    for (int i = 0; i < num_images; ++i){
        images_warped[i].convertTo(images_warped_f[i], CV_32F);
    }

}
    
    
void calculateCompensator(Ptr<ExposureCompensator> & compensator, vector<Point> corners, vector<UMat> images_warped, vector<UMat> & masks_warped, int num_images){

    logFile("Optimizando los bordes de las figuras...\n");



    //compensator = ExposureCompensator::createDefault(type); //cv::detail::BlocksGainCompensator

    if (!use_composer){
        compensator = ExposureCompensator::createDefault(ExposureCompensator::NO);
    }
    else{
        compensator = ExposureCompensator::createDefault(ExposureCompensator::GAIN_BLOCKS);
    }

    if (dynamic_cast<GainCompensator*>(compensator.get()))
    {
        GainCompensator* gcompensator = dynamic_cast<GainCompensator*>(compensator.get());
        gcompensator->setNrFeeds(1);
    }

    if (dynamic_cast<ChannelsCompensator*>(compensator.get()))
    {
        ChannelsCompensator* ccompensator = dynamic_cast<ChannelsCompensator*>(compensator.get());
        ccompensator->setNrFeeds(1);
    }

    if (dynamic_cast<BlocksCompensator*>(compensator.get()))
    {
        BlocksCompensator* bcompensator = dynamic_cast<BlocksCompensator*>(compensator.get());
        bcompensator->setNrFeeds(1);
        bcompensator->setNrGainsFilteringIterations(2);
        bcompensator->setBlockSize(32, 32);
    }

    compensator->feed(corners, images_warped, masks_warped);
    //masks_warped --> imagenes a actualizar en la funcion feed();



}


void estimateSeams(vector<UMat> images_warped_f, vector<Point> & corners, vector<UMat> & masks_warped, int num_images){

    logFile("Añadiendo textura...\n");
    
    Ptr<SeamFinder> seam_finder;
    #ifdef HAVE_OPENCV_CUDALEGACY
            if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
                seam_finder = makePtr<detail::GraphCutSeamFinderGpu>(GraphCutSeamFinderBase::COST_COLOR);
            else
    #endif
            seam_finder = makePtr<detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR);


    seam_finder->find(images_warped_f, corners, masks_warped);
    

}


//Utiliza las imágenes almacenadas en Carpeta
void composeImages(Ptr<Blender> & blender, int num_images, vector<int> indices, cv::Mat & full_img, vector<cv::Mat> & images, bool & is_compose_scale_set, double & compose_scale, double work_scale, float & warped_image_scale, Ptr<RotationWarper> & warper, Ptr<WarperCreator> warper_creator, vector<Size> full_img_sizes, vector<CameraParams> & cameras, vector<Point> & corners, vector<Size> & sizes, cv::Mat & img, vector<UMat> & masks_warped, Ptr<ExposureCompensator> & compensator){

    logFile("Uniendo frames...\n");

    Mat img_warped, img_warped_s;
    Mat dilated_mask, seam_mask, mask, mask_warped;
    double compose_work_aspect = 1;

    for (int img_idx = 0; img_idx < num_images; ++img_idx)
    {
        logFile("Uniendo frame #" + to_string(indices[img_idx]+1) + "\n");

        // Se lee la imagen, y se redimensiona si es necesario
        full_img = getFramei(folderName, img_idx);
        if (!is_compose_scale_set)
        {
            if (compose_megapix > 0)
                compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / full_img.size().area()));
            is_compose_scale_set = true;

            compose_work_aspect = compose_scale / work_scale;
            

            warped_image_scale *= static_cast<float>(compose_work_aspect);
            warper = warper_creator->create(warped_image_scale);


            // Se actualizan las esquinas y dimensiones
            for (int i = 0; i < num_images; ++i)
            {
                // Actualizar intrinsecos
                cameras[i].focal *= compose_work_aspect;
                cameras[i].ppx *= compose_work_aspect;
                cameras[i].ppy *= compose_work_aspect;

                Size sz = full_img_sizes[i];
                if (std::abs(compose_scale - 1) > 1e-1)
                {
                    sz.width = cvRound(full_img_sizes[i].width * compose_scale);
                    sz.height = cvRound(full_img_sizes[i].height * compose_scale);
                }

                Mat K;
                cameras[i].K().convertTo(K, CV_32F);
                Rect roi = warper->warpRoi(sz, K, cameras[i].R);
                corners[i] = roi.tl();
                sizes[i] = roi.size();
                
            }
        }
        if (abs(compose_scale - 1) > 1e-1)
            resize(full_img, img, Size(), compose_scale, compose_scale, INTER_LINEAR_EXACT);
        else
            img = full_img;
        full_img.release();
        Size img_size = img.size();

        Mat K;
        cameras[img_idx].K().convertTo(K, CV_32F);


        // Warp imagen
        warper->warp(img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

        // Warp mascara
        mask.create(img_size, CV_8U);
        mask.setTo(Scalar::all(255));
        warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);


        compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);

        img_warped.convertTo(img_warped_s, CV_16S);
        img_warped.release();
        img.release();
        mask.release();
  

        dilate(masks_warped[img_idx], dilated_mask, Mat());
        resize(dilated_mask, seam_mask, mask_warped.size(), 0, 0, INTER_LINEAR_EXACT);
        mask_warped = seam_mask & mask_warped;


        if (!blender)
        {
            blender = Blender::createDefault(Blender::MULTI_BAND, try_cuda);
            Size dst_sz = resultRoi(corners, sizes).size();
            float blend_width = sqrt(static_cast<float>(dst_sz.area())) * 5 / 100.f;
            if (blend_width < 1.f)
                blender = Blender::createDefault(Blender::NO, try_cuda);
                MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
                mb->setNumBands(static_cast<int>(ceil(log(blend_width)/log(2.)) - 1.));
            blender->prepare(corners, sizes);
        }

        // Unir imagenes (blend)
        blender->feed(img_warped_s, mask_warped, corners[img_idx]);
    }

}

////



static int parseCmdArgs(int argc, char** argv)
{
    if (argc == 1)
    {
        printUsage(argv);
        return -1;
    }
    for (int i = 1; i < argc; ++i)
    {
        if (string(argv[i]) == "--help" || string(argv[i]) == "/?")
        {
            printUsage(argv);
            return -1;
        }
        else if (string(argv[i]) == "--video")
        {
            if (argv[i + 1] != NULL){
                video_name = argv[i + 1];
                i++;
            }
            else{
                video_name = "";
            }
            
        }
        else if (string(argv[i]) == "--segundos")
        {
            if (argv[i + 1] != NULL){
                segundos = atoi(argv[i + 1]);
                i++;
            }
            else{
                segundos = -1;
            }

        }
        else if (string(argv[i]) == "--match_conf")
        {
            match_conf = static_cast<float>(atof(argv[i + 1]));
            i++;
        }
        else if (string(argv[i]) == "--compose_megapix")
        {
            compose_megapix = atof(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--use_composer")
        {
            use_composer = true;
        }
        else if (string(argv[i]) == "--output")
        {
            result_name = argv[i + 1];
            result_name = result_name + ".jpg";
            i++;
        }
    }
    return 0;
}


int main(int argc, char* argv[])
{

#if ENABLE_LOG

    auto start = std::chrono::system_clock::now();
    std::time_t start_time = std::chrono::system_clock::to_time_t(start);
    logFile("++++++++++++++++++++++++++++++++ Nueva Ejecución ++++++++++++++++++++++++++++++++\n");
    logFile("Fecha de ejecución: ");
    logFile(std::ctime(&start_time));
    logFile("\n\n\n");

#endif

#if 0
    cv::setBreakOnError(true);
#endif

    int retval = parseCmdArgs(argc, argv);
    if (retval)
        return retval;

    if (video_name == ""){
        std::cout << "Debe de proporcionarse un video [ --video <Nombre del Vídeo> ]" <<std::endl;
        return -1;
    }
    //system("exec mkdir usedFrames &> /dev/null");
    mkdir("usedFrames", S_IRWXU | S_IRWXG | S_IRWXO);


    int num_images = 0;
    num_images = initialFrames();
    logFile("Frames iniciales capturados: " + to_string(num_images) + "\n");

    //Se prueba a eliminar las imagenes cuyo overlap (superposición) con otras imágenes sea alto
    //imagenes que aportan poca informacion util
    //A mayor parámetro (match_conf) se eliminan más imágenes que aportan poca informacion

    vector<cv::Mat> frames;
    if (!use_composer){
        frames = keyFramesStoredSelector(folderName, num_images);
    }
    else{ //Se limitan los frames por falta de Hardware (RAM)
        int i = 0;
        for(; num_images > 130; i++){

            if(i > 0){ //No ha salido del bucle, se resta confianza
                logFile("Volviendo a recalcular frames clave modificando parámetros internos\n");
                cout << "Volviendo a recalcular frames clave modificando parámetros internos" << endl;

                if( (match_conf - 0.25) >= 0.35){
                    match_conf = match_conf - 0.25;
                    logFile("Modificado --> match_conf: " + to_string(match_conf) + "\n");
                }
                if( (segundos != -1) && ((segundos-5) > 0)){
                    segundos -= 5;
                    logFile("Modificado --> segundos: " + to_string(segundos) + "\n");
                }
                num_images = initialFrames();
            }
            frames = keyFramesStoredSelector(folderName, num_images);
        }
        if(i == 0){ //No ha entrado en el bucle for, se seleccionan claves sin modificar params
            frames = keyFramesStoredSelector(folderName, num_images);
        }
    }
    num_images = frames.size();


    system("exec rm -r ./usedFrames/*"); //BORRA todos los frames almacenados

    storeFrames(folderName, frames); //Guarda los frames en folder
    frames.clear();
    //getFramei(folderName, i);


    double work_scale = 1, seam_scale = 1, compose_scale = 1;
    bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;
    cv::Mat full_img, img;
    vector<ImageFeatures> features(num_images);
    vector<Mat> images(num_images);
    vector<Size> full_img_sizes(num_images);
    double seam_work_aspect = 1.0;


    vector<MatchesInfo> pairwise_matches;
    vector<int> indices;


    vector<CameraParams> cameras;


    vector<double> focals;


    float warped_image_scale;


    vector<Point> corners(num_images);
    vector<UMat> masks_warped(num_images);
    vector<UMat> images_warped(num_images);
    vector<Size> sizes(num_images);
    Ptr<WarperCreator> warper_creator;
    Ptr<RotationWarper> warper;
    vector<UMat> images_warped_f(num_images);


    Ptr<ExposureCompensator> compensator;


    Ptr<Blender> blender;
    
    //Uso los frames almacenados en la carpeta
    getFeatures(num_images, work_scale, seam_scale, is_work_scale_set, is_seam_scale_set, full_img, img, features, images, full_img_sizes, seam_work_aspect);
    //Liberar memoria
    full_img.release();
    img.release();
    getMatcher(pairwise_matches, features, indices, images, full_img_sizes);

    //El emparejador detecta menos frames de los clave seleccionados, eliminar los que no detecta
    if(indices.size() != num_images){
        vector<Mat> newImages;

        for(int i = indices[0]; i <= indices[indices.size()-1]; i++){ //Se procesan solo las que detecta el emparejador
            newImages.push_back( getFramei(folderName, i) );
        }
        system("exec rm -r ./usedFrames/*");
        logFile("Imagenes usadas en el emparejamiento --> " + to_string(newImages.size()) + "\n");
        storeFrames(folderName, newImages);
        //newImages.clear();

        //num_images = indices[indices.size()-1];
        num_images = newImages.size();
        newImages.clear();


        features.resize(num_images);
        images.resize(num_images);
        full_img_sizes.resize(num_images);
        corners.resize(num_images);
        masks_warped.resize(num_images);
        images_warped.resize(num_images);
        sizes.resize(num_images);
        images_warped_f.resize(num_images);

        getFeatures(num_images, work_scale, seam_scale, is_work_scale_set, is_seam_scale_set, full_img, img, features, images, full_img_sizes, seam_work_aspect);
        full_img.release();
        img.release();
        getMatcher(pairwise_matches, features, indices, images, full_img_sizes);
    }

    homographyEstimation(features, cameras, pairwise_matches, num_images);
    pairwise_matches.clear();
    cameraFocals(cameras, indices, focals);
    warpedImageScale(warped_image_scale, focals);
    warpImages(corners, masks_warped, images_warped, sizes, images, num_images, warper_creator, warper, warped_image_scale, seam_work_aspect, cameras, images_warped_f);
    //OutOfMemory Exception si el sistema no dispone de un minimo de RAM
    calculateCompensator(compensator, corners, images_warped, masks_warped, num_images);
    //Liberar memoria
    images_warped.clear();
    estimateSeams(images_warped_f, corners, masks_warped, num_images);
    //Liberar memoria
    images_warped_f.clear();
    //Uso los frames almacenados en la carpeta
    composeImages(blender, num_images, indices, full_img, images, is_compose_scale_set, compose_scale, work_scale, warped_image_scale, warper, warper_creator, full_img_sizes, cameras, corners, sizes, img, masks_warped, compensator);

    //Guarda el resultado final
    Mat result, result_mask;
    blender->blend(result, result_mask);
    imwrite(result_name, result);
    imwrite(result_name.substr(0, result_name.size()-4) + "_mask" + ".jpg", result_mask);

    cout << "Imágen resultante guardada con nombre..." << result_name << endl;
    cout << "Máscara resultante guardada con nombre..." << result_name.substr(0, result_name.size()-4) << "_mask" << ".jpg" << endl;
    logFile("Imágen resultante guardada con nombre..." + result_name + "\n");
    logFile("Máscara resultante guardada con nombre..." + result_name.substr(0, result_name.size()-4) + "_mask" + ".jpg" + "\n");

    images.clear();
    img_names.clear();

#if ENABLE_LOG
    auto end = std::chrono::system_clock::now();
    cout << "Tiempo de procesamiento: " << chrono::duration_cast<chrono::seconds>(end - start).count() << " segundos" << endl;
#endif

    return 0;
}
