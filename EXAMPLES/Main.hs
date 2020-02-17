import qualified TensorFlow.Core as TF
import qualified TensorFlow.GenOps.Core as TF 
import qualified TensorFlow.Gradient as TF
import qualified TensorFlow.Ops as TF

makeAxis :: [Float] -> [Float] -> [(Float, Float)]
makeAxis axis1 axis2 = [(t, t') | t <- axis1, t' <- axis2]

computeGd :: (Float, Float) -> IO (Float, Float, Float)
computeGd (x, y) = do
    let grads = do
                xv <- TF.render $ TF.scalar (x :: Float)
                yv <- TF.render $ TF.scalar (y :: Float)
                let xv2 = xv `TF.mul` xv
                    yv2 = yv `TF.mul` yv
                    r = TF.sqrt (xv2 + yv2)
                    fxy = r
                TF.gradients fxy [xv, yv]
    [dx, dy] <- TF.runSession $ grads >>= TF.run
    let 
       dxs = TF.unScalar dx 
       dys = TF.unScalar dy
       gdr = sqrt (dxs*dxs + dys*dys)
    -- print gdr
    return (x, y, gdr)

main :: IO ()
main = do
  let n = 1
      xs = (\x -> x / n) <$> [(- 5 * n :: Float) .. 5 * n]
      ys = (\x -> x / n) <$> [(- 5 * n :: Float) .. 5 * n]
      grid = makeAxis xs ys
  vectors <- mapM computeGd grid
  putStrLn "\nDone\n"
