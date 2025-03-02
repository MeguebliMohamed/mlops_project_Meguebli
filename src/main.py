import argparse
from src.model_pipeline import load_data_from_csv, upload_data_to_mysql, load_data_from_mysql, prepare_data, train_model, evaluate_model, save_model, load_model

def execute_command(command, csv_path=None, model_path=None, save_path=None):
    if command == "load_csv":
        data = load_data_from_csv(csv_path)
        if data is not None:
            print("\n✅ CSV data loaded successfully")
        return data
    
    elif command == "upload":
        data = execute_command("load_csv", csv_path=csv_path)
        if data is not None:
            upload_data_to_mysql(data)
    
    elif command == "load_data":
        data = load_data_from_mysql()
        if data is not None:
            print("\n✅ Data loaded from MySQL successfully")
        return data
    
    elif command == "prepare":
        data = execute_command("load_data")
        if data is not None:
            return prepare_data(data)
    
    elif command == "train":
        X_train, X_test, y_train, y_test, scaler, label_encoders = execute_command("prepare")
        if X_train is not None:
            model = train_model(X_train, y_train)
            print("\n✅ Model trained successfully")
            return model, X_test, y_test, scaler, label_encoders
    
    elif command == "evaluate":
        model, X_test, y_test, _, _ = execute_command("train")
        if model is not None:
            accuracy, report = evaluate_model(model, X_test, y_test)
            print(f"\n✅ Accuracy: {accuracy}\n{report}")
            return accuracy, report
    
    elif command == "save":
        if save_path:
            model, X_test, y_test, scaler, label_encoders = execute_command("train")
            if model is not None:
                save_model(model, scaler, label_encoders, save_path)
                print(f"\n✅ Model saved to {save_path}")
                return model, X_test, y_test, scaler, label_encoders
    
    elif command == "load":
        if model_path:
            model_data = load_model(model_path)
            print("\n✅ Model loaded successfully")
            return model_data
    
    elif command == "full_pipeline":
        execute_command("upload", csv_path=csv_path)
        model, X_test, y_test, scaler, label_encoders = execute_command("train")
        if model is not None:
            save_model(model, scaler, label_encoders, save_path or "model.joblib")
            accuracy, report = evaluate_model(model, X_test, y_test)
            print(f"\n✅ Full pipeline completed. Accuracy: {accuracy}\n{report}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLOps Pipeline")
    parser.add_argument("command", type=str, help="Command: load_csv, upload, load_data, prepare, train, evaluate, save, load, full_pipeline")
    parser.add_argument("--csv", type=str, help="Path to CSV file", default="data/data.csv")
    parser.add_argument("--load", type=str, help="Path to load a model")
    parser.add_argument("--save", type=str, help="Path to save the model", default="model.joblib")
    args = parser.parse_args()
    execute_command(args.command, csv_path=args.csv, model_path=args.load, save_path=args.save)