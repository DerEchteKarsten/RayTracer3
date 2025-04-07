use anyhow::{Result, Error};

use super::RenderGraph;

impl RenderGraph {
    fn run(&self) -> Result<()> {
        let output = self.output.ok_or(Error::msg("No Output set".to_string()))?;
        let execution_order = vec![vec![]];
        
        let root_node = self.graph.iter().find(|e| e.writes.contains(&output)).ok_or("No Node writes to output".to_string())?;
        
        


        Ok(())
    }
}
