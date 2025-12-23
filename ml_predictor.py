"""
CrowdAID - ML Model Predictor
Menggunakan trained Random Forest model untuk prediksi
"""

import pandas as pd
import pickle
import json

class CrowdAIDPredictor:
    """
    Predictor class untuk CrowdAID menggunakan trained ML model
    """
    
    def __init__(self, model_path='model_random_forest.pkl', 
                 encoders_path='label_encoders.pkl',
                 metadata_path='model_metadata.json'):
        """
        Initialize predictor dengan loading model dan encoders
        """
        # Load model
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load encoders
        with open(encoders_path, 'rb') as f:
            self.encoders = pickle.load(f)
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        print("✅ Model loaded successfully!")
        print(f"   Model accuracy: {self.metadata['accuracy_train']*100:.2f}%")
    
    def predict_suitability(self, hospital_type, hospital_class, 
                          capacity, services, staff, condition):
        """
        Predict apakah hospital cocok untuk condition tertentu
        
        Returns:
            - probability: float (0-1)
            - is_suitable: bool
            - confidence: str (High/Medium/Low)
        """
        # Encode inputs
        try:
            type_encoded = self.encoders['hospital_type'].transform([hospital_type])[0]
            class_encoded = self.encoders['hospital_class'].transform([hospital_class])[0]
            condition_encoded = self.encoders['condition'].transform([condition])[0]
        except:
            # If unknown value, return not suitable
            return {
                'probability': 0.0,
                'is_suitable': False,
                'confidence': 'Unknown',
                'score': 0
            }
        
        # Create feature vector
        features = [[type_encoded, class_encoded, capacity, services, 
                    staff, condition_encoded]]
        
        # Predict
        probability = self.model.predict_proba(features)[0][1]  # Probability of being suitable
        is_suitable = probability >= 0.5
        
        # Confidence level
        if probability >= 0.8 or probability <= 0.2:
            confidence = 'High'
        elif probability >= 0.6 or probability <= 0.4:
            confidence = 'Medium'
        else:
            confidence = 'Low'
        
        # Convert to AI score (0-100)
        score = int(probability * 100)
        
        return {
            'probability': probability,
            'is_suitable': is_suitable,
            'confidence': confidence,
            'score': score
        }
    
    def get_recommendations(self, hospitals_df, condition, location=None):
        """
        Get ranked recommendations untuk kondisi tertentu
        
        Args:
            hospitals_df: DataFrame dengan hospital data
            condition: str - kondisi pasien
            location: str - kabupaten/kota (optional)
        
        Returns:
            DataFrame dengan ranked recommendations
        """
        recommendations = []
        
        # Filter by location if specified
        if location:
            hospitals_df = hospitals_df[hospitals_df['kab'] == location]
        
        # Predict for each hospital
        for idx, row in hospitals_df.iterrows():
            result = self.predict_suitability(
                hospital_type=row['jenis'],
                hospital_class=row['kelas'],
                capacity=row['total_tempat_tidur'],
                services=row['total_layanan'],
                staff=row['total_tenaga_kerja'],
                condition=condition
            )
            
            if result['is_suitable']:
                recommendations.append({
                    'nama': row['nama'],
                    'alamat': row['alamat'],
                    'jenis': row['jenis'],
                    'kelas': row['kelas'],
                    'kapasitas': row['total_tempat_tidur'],
                    'layanan': row['total_layanan'],
                    'staff': row['total_tenaga_kerja'],
                    'ml_score': result['score'],
                    'probability': result['probability'],
                    'confidence': result['confidence']
                })
        
        # Convert to DataFrame and sort by score
        if recommendations:
            df_recommendations = pd.DataFrame(recommendations)
            df_recommendations = df_recommendations.sort_values('ml_score', ascending=False)
            return df_recommendations
        else:
            return pd.DataFrame()
    
    def get_feature_importance(self):
        """
        Get feature importance dari model
        """
        return pd.DataFrame(self.metadata['feature_importance'])


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    print("="*70)
    print("CrowdAID ML PREDICTOR - DEMO")
    print("="*70)
    
    # Initialize predictor
    predictor = CrowdAIDPredictor()
    
    # Load hospital data
    df_hospitals = pd.read_csv('/mnt/user-data/uploads/Hospital_Banten.csv', sep=';')
    
    # Example 1: Predict untuk kondisi Kebidanan
    print("\n" + "="*70)
    print("EXAMPLE 1: Rekomendasi untuk Kebidanan di Kota Tangerang Selatan")
    print("="*70)
    
    recommendations = predictor.get_recommendations(
        hospitals_df=df_hospitals,
        condition='Kebidanan',
        location='Kota Tangerang Selatan'
    )
    
    print(f"\n✅ Found {len(recommendations)} suitable hospitals")
    if len(recommendations) > 0:
        print("\nTop 5 Recommendations:")
        print(recommendations.head(5)[['nama', 'jenis', 'kelas', 'ml_score', 'confidence']].to_string(index=False))
    
    # Example 2: Predict untuk kondisi Banyak Spesialis
    print("\n" + "="*70)
    print("EXAMPLE 2: Rekomendasi untuk Banyak Spesialis di Kota Tangerang")
    print("="*70)
    
    recommendations = predictor.get_recommendations(
        hospitals_df=df_hospitals,
        condition='Banyak Spesialis',
        location='Kota Tangerang'
    )
    
    print(f"\n✅ Found {len(recommendations)} suitable hospitals")
    if len(recommendations) > 0:
        print("\nTop 5 Recommendations:")
        print(recommendations.head(5)[['nama', 'jenis', 'kelas', 'ml_score', 'confidence']].to_string(index=False))
    
    # Example 3: Single hospital prediction
    print("\n" + "="*70)
    print("EXAMPLE 3: Single Hospital Prediction")
    print("="*70)
    
    test_hospital = df_hospitals.iloc[0]
    print(f"\nHospital: {test_hospital['nama']}")
    print(f"Type: {test_hospital['jenis']}")
    print(f"Class: {test_hospital['kelas']}")
    
    # Test dengan berbagai kondisi
    conditions = ['Gejala Ringan', 'Penyakit Dalam', 'Kebidanan', 'Banyak Spesialis']
    print("\nSuitability for different conditions:")
    for cond in conditions:
        result = predictor.predict_suitability(
            hospital_type=test_hospital['jenis'],
            hospital_class=test_hospital['kelas'],
            capacity=test_hospital['total_tempat_tidur'],
            services=test_hospital['total_layanan'],
            staff=test_hospital['total_tenaga_kerja'],
            condition=cond
        )
        print(f"  {cond:20s}: Score={result['score']:3d}/100  Suitable={result['is_suitable']}  Confidence={result['confidence']}")
    
    # Feature importance
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE")
    print("="*70)
    importance = predictor.get_feature_importance()
    print(importance.to_string(index=False))
