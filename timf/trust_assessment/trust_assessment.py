

class TrustAssessment:
    def __init__(self,weight_matrix):
        self.weight_matrix = weight_matrix
        


    def calculate(self,datain):
            datain['trust_score'] = (
            datain['speed'] * self.weight_matrix[0] +
            datain['latency'] * self.weight_matrix[1] +
            datain['bandwidth'] * self.weight_matrix[2] +
            datain['coverage'] * self.weight_matrix[3] +
            datain['reliability'] * self.weight_matrix[4] +
            datain['security'] * self.weight_matrix[5]) 
            return datain['trust_score'].mean()
            
        
        
    