from deepface import DeepFace


def real_time(db_path, model_name, distance_metric, enable_face_analysis=False, detector_backend='mtcnn'):
    """
        Function to initiate real-time face recognition using DeepFace.

        Parameters:
        db_path (str): Path to the directory containing the database of known faces.
        model_name (str): Name of the model to use for face recognition (e.g., 'Facenet').
        distance_metric (str): Distance metric to use for face comparison (e.g., 'euclidean_l2').
        enable_face_analysis (bool): Whether to enable additional face analysis (age, gender, emotion) (default is False).
        detector_backend (str): Face detector backend to use (e.g., 'mtcnn').
    """
    DeepFace.stream(
        db_path=db_path,
        model_name=model_name,
        distance_metric=distance_metric,
        enable_face_analysis=enable_face_analysis,
        detector_backend=detector_backend,
        source=0
    )


if __name__ == '__main__':
    '''
    The embedding data will be saved as embedding.pkl in the db_path directory. 
    This allows the data to be loaded directly in the future, eliminating the need for recalculation.
    '''
    db_path = r'../data/I-500'
    model_name = 'Facenet'
    distance_metric = 'euclidean_l2'
    enable_face_analysis = False
    detector_backend = 'mtcnn'

    # Start the real-time face recognition stream
    real_time(
        db_path=db_path,
        model_name=model_name,
        distance_metric=distance_metric,
        enable_face_analysis=enable_face_analysis,
        detector_backend=detector_backend
    )
