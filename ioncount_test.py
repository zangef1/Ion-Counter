import MDAnalysis as mda
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from dataclasses import dataclass
import warnings
import json

@dataclass
class Surface:
    residue_indices: List[int]
    center: np.ndarray = None

class Channel:
    def __init__(self, radius: float):
        self.radius = radius
        self.axis_vector = None
        self.axis_center = None
        self.length = None

class IonPermeationAnalysis:
    def __init__(self, 
                 universe: mda.Universe,
                 ion_selection: str,
                 upper_gate_residues: List[int],  # Surface 1: upper gate
                 lower_gate_residues: List[int],  # Surface 2: lower gate
                 channel_radius: float = 5.0,
                 min_frame_gap: int = 10):
        
        self.u = universe
        self.ion_selection = ion_selection
        self.min_frame_gap = min_frame_gap
        
        # Initialize gates (surfaces)
        self.upper_gate = Surface(upper_gate_residues)  # Surface 1
        self.lower_gate = Surface(lower_gate_residues)  # Surface 2
        self.channel = Channel(channel_radius)
        
        # Data storage
        self.permeation_events = []
        self.ion_positions = {}
        self.last_crossing_frames = {}
        self.gate_positions = []  # Store gate positions over time
        
    def _update_gate_positions(self, ts):
        """Update gate positions for current frame."""
        # Select atoms for upper gate (Surface 1)
        upper_sel = self.u.select_atoms(f"resid {' '.join(map(str, self.upper_gate.residue_indices))}")
        if len(upper_sel) == 0:
            raise ValueError("No atoms found for upper gate")
        
        # Select atoms for lower gate (Surface 2)
        lower_sel = self.u.select_atoms(f"resid {' '.join(map(str, self.lower_gate.residue_indices))}")
        if len(lower_sel) == 0:
            raise ValueError("No atoms found for lower gate")
        
        # Calculate centers of mass
        self.upper_gate.center = upper_sel.center_of_mass()
        self.lower_gate.center = lower_sel.center_of_mass()
        
        # Calculate channel axis for this frame
        axis_vector = self.lower_gate.center - self.upper_gate.center  # Direction from upper to lower
        self.channel.length = np.linalg.norm(axis_vector)
        self.channel.axis_vector = axis_vector / self.channel.length
        self.channel.axis_center = (self.upper_gate.center + self.lower_gate.center) / 2
        
        # Store gate positions for this frame
        self.gate_positions.append({
            'frame': int(ts.frame),
            'upper_gate_pos': [float(x) for x in self.upper_gate.center],
            'lower_gate_pos': [float(x) for x in self.lower_gate.center],
            'channel_length': float(self.channel.length)
        })
        
        if ts.frame % 100 == 0:
            print(f"\nFrame {ts.frame} gate positions:")
            print(f"Upper gate: {self.upper_gate.center}")
            print(f"Lower gate: {self.lower_gate.center}")
            print(f"Channel length: {self.channel.length:.2f} Å")

    def _point_in_channel(self, point: np.ndarray) -> Tuple[bool, float, float]:
        """
        Check if a point is within the cylindrical channel.
        Returns: (is_inside, radial_distance, axial_position)
        """
        # Vector from channel center to point
        v = point - self.channel.axis_center
        
        # Project onto axis
        axial_pos = np.dot(v, self.channel.axis_vector)
        
        # Get radial component
        proj = axial_pos * self.channel.axis_vector
        radial_vec = v - proj
        radial_dist = np.linalg.norm(radial_vec)
        
        # Check if within channel
        is_inside = (radial_dist <= self.channel.radius and 
                    abs(axial_pos) <= self.channel.length/2)
        
        return is_inside, float(radial_dist), float(axial_pos)

    def analyze_frame(self, ts):
        """Analyze a single trajectory frame with dynamic gate positions."""
        try:
            # Update gate positions for this frame
            self._update_gate_positions(ts)
            
            # Select all K+ ions
            ions = self.u.select_atoms(self.ion_selection)
            
            # Print total ions at first frame
            if ts.frame == 0:
                print(f"\nTotal number of {self.ion_selection} ions: {len(ions)}")
                print(f"Ion indices: {[ion.index for ion in ions]}")
            
            for ion in ions:
                ion_id = ion.index
                pos = ion.position
                
                # Check if ion is in channel
                is_inside, r_dist, z_pos = self._point_in_channel(pos)
                
                # Store position for ALL ions
                if ion_id not in self.ion_positions:
                    self.ion_positions[ion_id] = []
                
                self.ion_positions[ion_id].append({
                    'frame': int(ts.frame),
                    'position': [float(x) for x in pos],
                    'in_channel': bool(is_inside),
                    'radial_dist': float(r_dist),
                    'axial_pos': float(z_pos),
                    'dist_to_upper': float(np.linalg.norm(pos - self.upper_gate.center)),
                    'dist_to_lower': float(np.linalg.norm(pos - self.lower_gate.center))
                })
                
                # Check for permeation
                if is_inside or abs(z_pos) <= self.channel.length:
                    self._check_permeation(ion_id, z_pos, ts.frame)
                    
        except Exception as e:
            print(f"Warning: Error processing frame {ts.frame}: {str(e)}")

    def _check_permeation(self, ion_id: int, z_pos: float, frame: int):
        """Check for permeation events with dynamic gate positions."""
        if ion_id not in self.last_crossing_frames:
            self.last_crossing_frames[ion_id] = frame
            return
            
        # Check frame gap since last crossing
        if frame - self.last_crossing_frames[ion_id] < self.min_frame_gap:
            return
            
        # Get previous position
        prev_positions = self.ion_positions[ion_id]
        if len(prev_positions) < 2:
            return
            
        prev_z = prev_positions[-2]['axial_pos']
        
        # Check for crossing through channel
        # Using quarter points of current channel length
        if (prev_z < -self.channel.length/4 and z_pos > self.channel.length/4):
            self.permeation_events.append({
                'ion_id': str(ion_id),
                'frame': int(frame),
                'direction': 'forward',
                'channel_length': float(self.channel.length)
            })
            self.last_crossing_frames[ion_id] = frame
            print(f"\nForward permeation detected:")
            print(f"Ion {ion_id} at frame {frame}")
            print(f"Previous z: {prev_z:.2f}, Current z: {z_pos:.2f}")
            
        elif (prev_z > self.channel.length/4 and z_pos < -self.channel.length/4):
            self.permeation_events.append({
                'ion_id': str(ion_id),
                'frame': int(frame),
                'direction': 'backward',
                'channel_length': float(self.channel.length)
            })
            self.last_crossing_frames[ion_id] = frame
            print(f"\nBackward permeation detected:")
            print(f"Ion {ion_id} at frame {frame}")
            print(f"Previous z: {prev_z:.2f}, Current z: {z_pos:.2f}")

    def run_analysis(self):
        """Run analysis on entire trajectory."""
        print(f"\nStarting analysis of {len(self.u.trajectory)} frames...")
        start_time = datetime.now()
        
        for ts in self.u.trajectory:
            if ts.frame % 100 == 0:
                print(f"Processing frame {ts.frame}/{len(self.u.trajectory)}")
            self.analyze_frame(ts)
            
        print(f"\nAnalysis completed in {datetime.now() - start_time}")

    def get_results(self) -> Dict:
        """Get analysis results with ion statistics."""
        total_frames = int(len(self.u.trajectory))
        
        # Track all ions' positions relative to channel
        ion_statistics = {}
        for ion_id, positions in self.ion_positions.items():
            ion_statistics[str(ion_id)] = {  # Convert ion_id to string
                'frames_tracked': int(len(positions)),
                'frames_in_channel': int(sum(1 for p in positions if p['in_channel'])),
                'times_near_upper': int(sum(1 for p in positions 
                    if -self.channel.length/2 <= p['axial_pos'] <= -self.channel.length/4)),
                'times_near_lower': int(sum(1 for p in positions 
                    if self.channel.length/4 <= p['axial_pos'] <= self.channel.length/2)),
                'min_z': float(min(p['axial_pos'] for p in positions)),
                'max_z': float(max(p['axial_pos'] for p in positions))
            }
        
        # Format permeation events
        formatted_events = []
        for event in self.permeation_events:
            formatted_events.append({
                'ion_id': str(event['ion_id']),
                'frame': int(event['frame']),
                'direction': str(event['direction']),
                'channel_length': float(event['channel_length'])
            })
        
        forward_events = int(sum(1 for e in formatted_events if e['direction'] == 'forward'))
        backward_events = int(sum(1 for e in formatted_events if e['direction'] == 'backward'))
        
        # Gate dynamics statistics
        channel_lengths = [float(p['channel_length']) for p in self.gate_positions]
        
        return {
            'total_frames': total_frames,
            'total_ions_tracked': len(ion_statistics),
            'total_permeations': len(formatted_events),
            'forward_permeations': forward_events,
            'backward_permeations': backward_events,
            'permeation_rate_per_frame': float(len(formatted_events)) / float(total_frames),
            'channel_properties': {
                'average_length': float(np.mean(channel_lengths)),
                'min_length': float(np.min(channel_lengths)),
                'max_length': float(np.max(channel_lengths)),
                'radius': float(self.channel.radius)
            },
            'permeation_events': formatted_events,
            'ion_statistics': ion_statistics
        }

    def save_results(self, output_dir: str = 'results'):
        """Save analysis results with detailed gate dynamics."""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        results = self.get_results()
        
        # Save main results
        with open(output_dir / 'permeation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save gate dynamics
        with open(output_dir / 'gate_dynamics.json', 'w') as f:
            json.dump({
                'gate_positions': self.gate_positions,
                'sampling_info': {
                    'total_frames': int(len(self.u.trajectory)),
                    'time_between_frames': float(self.u.trajectory.dt)
                }
            }, f, indent=2)
        
        # Save readable summary
        with open(output_dir / 'analysis_summary.txt', 'w') as f:
            f.write(f"Analysis Summary\n{'='*50}\n\n")
            
            # Channel Properties
            f.write("Channel Properties:\n")
            f.write(f"Average length: {results['channel_properties']['average_length']:.2f} Å\n")
            f.write(f"Length range: {results['channel_properties']['min_length']:.2f} - ")
            f.write(f"{results['channel_properties']['max_length']:.2f} Å\n")
            f.write(f"Radius: {results['channel_properties']['radius']:.2f} Å\n\n")
            
            # Permeation Statistics
            f.write("Permeation Statistics:\n")
            f.write(f"Total ions tracked: {results['total_ions_tracked']}\n")
            f.write(f"Total permeation events: {results['total_permeations']}\n")
            f.write(f"Forward permeations: {results['forward_permeations']}\n")
            f.write(f"Backward permeations: {results['backward_permeations']}\n")
            f.write(f"Permeation rate: {results['permeation_rate_per_frame']:.6f} events/frame\n\n")
            
            # Gate dynamics
            f.write("\nGate Dynamics:\n")
            f.write(f"Maximum channel opening: {results['channel_properties']['max_length']:.2f} Å\n")
            f.write(f"Minimum channel opening: {results['channel_properties']['min_length']:.2f} Å\n")
            f.write(f"Average channel opening: {results['channel_properties']['average_length']:.2f} Å\n")

def main():
    """Main function to run the analysis."""
    top_file = "/home/data/Yongcheng/GIRK4/CHL/com_4fs.prmtop"
    traj_file = "/home/data/Yongcheng/GIRK4/CHL/G4_CHL_PIP2_1us.trr"
    
    # Define gate residues
    upper_gate_residues = [138, 788, 463, 1113]  # Surface 1
    lower_gate_residues = (
        list(range(259, 266)) +  # 259-265
        list(range(909, 916)) +  # 909-915
        list(range(584, 591)) +  # 584-590
        list(range(1234, 1241))  # 1234-1240
    )
    
    try:
        # Load system
        print(f"Loading system...")
        print(f"Topology file: {top_file}")
        print(f"Trajectory file: {traj_file}")
        u = mda.Universe(top_file, traj_file)
        
        # Create analyzer
        analyzer = IonPermeationAnalysis(
            universe=u,
            ion_selection="resname K+",
            upper_gate_residues=upper_gate_residues,
            lower_gate_residues=lower_gate_residues,
            channel_radius=5.0,    # Angstroms
            min_frame_gap=10       # Minimum frames between permeation events
        )
        
        # Run analysis
        analyzer.run_analysis()
        
        # Save results
        analyzer.save_results()
        
        # Print summary
        results = analyzer.get_results()
        print("\nAnalysis Summary:")
        print(f"Total permeation events: {results['total_permeations']}")
        print(f"Forward permeations: {results['forward_permeations']}")
        print(f"Backward permeations: {results['backward_permeations']}")
        print(f"Permeation rate: {results['permeation_rate_per_frame']:.6f} events/frame")
        print("\nResults have been saved to the 'results' directory")
        
    except Exception as e:
        print(f"Error running analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()
    
    
