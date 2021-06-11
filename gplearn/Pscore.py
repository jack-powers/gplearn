import simpy
import random
# import matplotlib.pyplot as plt
import numpy as np
# import plotly.graph_objects as go
import pandas as pd

random.seed(42)
GRAPH_FLAG = False
PRINT_FLAG = False



class GLOBALS:

    # Set up counter for number for patients entering simulation
    patient_count = 0
    rec_wait_times = [20, 50, 100]
    waiting_list_recalc_time = 2

    # Lists used to store audit results
    audit_time = []
    audit_patients_in_surgery = []
    audit_patients_waiting = []    
    audit_resources_used = []


    # for in turn stats
    patient_entering_system_ID = []
    patient_entering_surgery_ID = []
    patients_treated_in_turn = 0
    patients_treated_out_turn = 0
    
    patients_treated_in_time = 0
    patients_treated_out_time = 0

    # Set up running counts of patients waiting (total and by priority)
    patients_waiting = 0
    patients = []
    
    functional_form = None

class _Function(object):

    """A representation of a mathematical relationship, a node in a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting vector based on a mathematical relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(x1, *args) that returns a Numpy
        array of the same shape as its arguments.

    name : str
        The name for the function as it should be represented in the program
        and its visualizations.

    arity : int
        The number of arguments that the ``function`` takes.

    """

    def __init__(self, function, name, arity):
        self.function = function
        self.name = name
        self.arity = arity

    def __call__(self, *args):
        return self.function(*args)

class Patient:
    def __init__(self,env, OR_Rooms, func_form):
        GLOBALS.patient_count += 1
        self.env = env
        self.ID = GLOBALS.patient_count
        self.OR_Rooms = OR_Rooms
        #self.priority = random.randint(1,3)
        
        
        GLOBALS.patient_entering_system_ID.append(self.ID)
        
        self.URG_category = random.randint(1,3)
        self.max_wait_time = GLOBALS.rec_wait_times[self.URG_category - 1]
        
        
        self.s = random.uniform(0,1) * 4
        self.p = random.uniform(0,1) * 4
        self.r = random.uniform(0,1) * 4
        
        clinical_factors = [self.s, self.p, self.r] # equiv to X array in gplearn
        # self.l = random.uniform(0,1) * 4
        # self.i = random.uniform(0,1) * 4
        # self.d = random.uniform(0,1) * 4
        # self.c = random.uniform(0,1) * 4
        # self.w = random.uniform(0,1) * 4
        
        
        self.p_score_coefficent = self.calculate_coefficient(func_form, clinical_factors)###3*(0.4*self.r ** 2 + 0.4*self.s ** 2 + 0.2*self.p ** 2) #if self.r > 2 else (1+0.23*self.s**2+0.14*self.p**2+0.15*self.r**2+0.14*self.l**2+0.12*self.i**2+0.05*self.d**2+0.08*self.c**2+0.09*self.w**2)               
        self.p_score = round(self.p_score_coefficent * -1)
        print("done coef")
    
        
        self.surgery_length = 20     
        self.queuing_time = 0 # initially zero queuing time (changed later if required)
        
        self.time_in = env.now # time patient enters waiting list
        
        self.surgery_start = 0 # time patient enters surgery
        self.time_out = 0 # time patient leaves surgery
        
        self.p_score_update_times = [self.time_in] # P-score update over time
        self.p_score_updates = [self.p_score]
        
        self.patient_active = True
        self.surgery_complete = False
        self.patient_treated_in_time = -1
        
        self.patient_rank = [] # patients rank compared to all others at point in time
        self.patient_rank_time = []            
        
        
        
        GLOBALS.patients_waiting += 1
        
        self.env.process(self.patient_process())
        self.env.process(self.calculate_rank())
        
            
    def calculate_coefficient(self, functional_form, X):
            """Execute the program according to X.

            Parameters
            ----------
            X : {array-like}, shape = [n_samples, n_features]
                Training vectors, where n_samples is the number of samples and
                n_features is the number of features.

            Returns
            -------
            y_hats : array-like, shape = [n_samples]
                The result of executing the program on X.

            """
            # Check for single-node programs
            node = functional_form.program[0]
            if isinstance(node, float):
                return np.repeat(node, X.shape[0])
            if isinstance(node, int):
                return X[:, node]

            apply_stack = []

            for node in functional_form.program:

                if isinstance(node, _Function):
                    apply_stack.append([node])
                else:
                    # Lazily evaluate later
                    apply_stack[-1].append(node)

                while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
                    # Apply functions that have sufficient arguments
                    function = apply_stack[-1][0]
                    terminals = [np.repeat(t, X.shape[0]) if isinstance(t, float)
                                else X[:, t] if isinstance(t, int)
                                else t for t in apply_stack[-1][1:]]
                    intermediate_result = function(*terminals)
                    if len(apply_stack) != 1:
                        apply_stack.pop()
                        apply_stack[-1].append(intermediate_result)
                    else:
                        return intermediate_result

            # We should never get here
            return None

        
        
    def calculate_rank(self):
        while True:
            yield self.env.timeout(1)
            
            GLOBALS.audit_time.append(self.env.now)
            GLOBALS.audit_resources_used.append(self.OR_Rooms.count)
            
            if self.patient_active:          
                    
                patient_IDs = []
                patient_p_scores = []            
                
                for patient in GLOBALS.patients:
                    if patient.patient_active:
                        patient_IDs.append(patient.ID)
                        patient_p_scores.append(patient.p_score)
                
                idx = np.argsort(patient_p_scores)
                
                patient_IDs = np.array(patient_IDs)
                
                ID_idx_acs = patient_IDs[idx]
                
                rank = np.where(ID_idx_acs == self.ID)
                
                rank = (np.asarray(rank)).tolist()
                rank = rank[0][0]
                rank += 1
                self.patient_rank.append(rank)
                self.patient_rank_time.append(self.env.now)                        
                    

    def patient_process(self):
        with self.OR_Rooms.request(priority=self.p_score) as req:
            
            results = yield req | self.env.timeout(GLOBALS.waiting_list_recalc_time)

            if req in results: #patient made it through without being recalcaulted
                GLOBALS.patient_entering_surgery_ID.append(self.ID)
                
                entering_system_rank = GLOBALS.patient_entering_system_ID.index(self.ID) + 1
                entering_surgery_rank = GLOBALS.patient_entering_surgery_ID.index(self.ID) + 1
                
                if entering_system_rank >= entering_surgery_rank:
                    GLOBALS.patients_treated_in_turn += 1
                else:
                    GLOBALS.patients_treated_out_turn += 1               
                
                self.patient_active = False
                                                        
                self.queuing_time = self.env.now - self.time_in
                if self.queuing_time < self.max_wait_time:
                    GLOBALS.patients_treated_in_time += 1
                    self.patient_treated_in_time = 1
                else:
                    GLOBALS.patients_treated_out_time += 1
                    self.patient_treated_in_time = 0
                
                
                self.surgery_start = self.env.now
                
                if PRINT_FLAG:
                    print("%2.2f Patient %s(%d) entering surgery. Waited %2.2f" % (self.env.now, self.ID, self.p_score, self.queuing_time))   
                
                yield self.env.timeout(self.surgery_length)
                
                if PRINT_FLAG:
                    print("%2.2f Patient %s(%d) completed surgery" % (self.env.now, self.ID, self.p_score))

                self.time_out = self.env.now
                
                self.surgery_complete = True
                
                
                
                self.p_score_update_times.append(self.env.now)
                self.p_score_updates.append(self.p_score)  
                GLOBALS.patients_waiting -+ 1
                            
            else: # patient spot being recalualted                
                self.p_score = round(self.p_score_coefficent * -1 * (self.env.now - self.time_in))                    
                self.p_score_update_times.append(self.env.now)
                self.p_score_updates.append(self.p_score)                                           
                self.env.process(self.patient_process())   
                
            
    
  
###### Patient Creation #######

def Model(functonal_form):
    env = simpy.Environment()

    OR_rooms = simpy.PriorityResource(env, 2)
    # patients = []

    for _ in range(20):
        GLOBALS.patients.append(Patient(env, OR_rooms,functonal_form))


    def patients_source(env):
        while True:
            yield env.timeout(random.expovariate(1/11))  # timeout before creation
            p = Patient(env, OR_rooms,functonal_form)
            GLOBALS.patients.append(p)
        
    
    env.process(patients_source(env))  ## continuous patient source
    #env.process(outside_rank())

    env.run(until=3000)



    patient_data = pd.DataFrame(columns=['p_score','p_score_coefficient', 'q_time', 'surgery_length', 'system_time', 'treated_in_time'])

    for patient in GLOBALS.patients:
        if patient.surgery_complete:
            patient_data.loc[patient.ID] = [patient.p_score_updates[-1], patient.p_score_coefficent, patient.queuing_time, patient.surgery_length, patient.queuing_time+patient.surgery_length, patient.patient_treated_in_time]



    # print(patient_data)
    # print(patient_data.describe())
    
    in_turn = GLOBALS.patients_treated_in_turn*100 / (GLOBALS.patients_treated_in_turn + GLOBALS.patients_treated_out_turn)
    in_time = GLOBALS.patients_treated_in_time*100 / (GLOBALS.patients_treated_in_time + GLOBALS.patients_treated_out_time)
    

    # print("Patients treated in turn: %d" % GLOBALS.patients_treated_in_turn)
    # print("Treated in turn %%: %2.2f%%" % in_turn)
    # print("Treated in time %%: %2.2f%%" % in_time)
    # print("Average resources used: %2.2f" % (np.average(GLOBALS.audit_resources_used)))
    
    return in_turn, in_time

# in_turn, in_time = Model()
