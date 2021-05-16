import Control.Arrow
import System.Directory
import Data.Char
import qualified Data.Text as T
import qualified Data.Text.IO as T
import Data.List
import qualified Data.Set as S
import qualified Data.Map.Strict as M
import qualified Data.IntMap.Strict as IM
import NLP.Stemmer
import Data.Ord
import Data.Function
import Data.SVM
import qualified Numeric.LinearAlgebra.Data as LA
import qualified Numeric.LinearAlgebra as LA

common_threshold = 200

-- Ratings are embedded in the file names, e.g., 100_7.txt has a rating of 7/10. Extract it.
getRating :: (Fractional a, Read a) => String -> a
getRating = (\x -> (x-5.5)/4.5) . read . takeWhile (/= '.') . drop 1 . dropWhile (/= '_')

-- Get the rating and text from all files in a directory.
readSamples :: (Fractional a, Read a) => String -> IO [(a, T.Text)]
readSamples dir = mapM ((sequence .) $ getRating &&& T.readFile . (dir++)) . filter ((/= '.') . head) =<< getDirectoryContents dir

-- All of the training data, which is located across two directories.
trainingSamples :: (Fractional a, Read a) => IO [(a, T.Text)]
validationSamples :: (Fractional a, Read a) => IO [(a, T.Text)]
--trainingSamples = (++) <$> readSamples "aclImdb/train/pos/" <*> readSamples "aclImdb/train/neg/"
trainingSamples = do
  pos <- readSamples "aclImdb/train/pos/"
  neg <- readSamples "aclImdb/train/neg/"
  return $ takeEvery 5 (pos ++ neg)

validationSamples = do
  pos <- readSamples "aclImdb/train/pos/"
  neg <- readSamples "aclImdb/train/neg/"
  return $ takeEvery 10 (undefined:(pos ++ neg))

-- Words to ignore
stoplist :: S.Set T.Text
stoplist = S.fromList $ map T.pack ["i", "me", "my", "she", "her", "he", "him", "his", "they", "them", "their", "it", "who", "this", "that", "which",
  "", "br", "the", "a", "an", "is", "was", "be", "have", "are", "had", "were", "do", "did",
  "and", "or", "to", "in", "for", "with", "as", "one", "at", "by", "about", "from", "other", "into", "when", "would", "than",
  "also", "how", "becaus", "could", "after", "thing", "dont", "year", "two", "after", "peopl", "get", "where", "did", "off"]

minimalStoplist = S.fromList $ map T.pack ["the", "a", "an", "to", "is", "that", ""]
negateList = S.fromList $ map T.pack ["not", "isnt"]

-- Word splitter which ignores puctuation and doesn't care about capitalization or apostrophes.
words' :: T.Text -> [T.Text]
words' = map (stem' English) . filter (not . T.null) . T.split (not . isAlpha) . T.toLower . T.filter (/= '\'')

-- Apply +/- modifier to word list
preFilter :: Num a => [T.Text] -> [(a, T.Text)]
preFilter [] = []
preFilter (w:ws)
  | w `S.member` negateList = alterFirst (first negate) (preFilter ws)
  | w `S.member` minimalStoplist = preFilter ws
  | T.pack "un" `T.isPrefixOf` w = (-1, T.drop 2 w) : preFilter ws
  | T.pack "nt" `T.isSuffixOf` w = (-1, T.dropEnd 2 w) : preFilter ws
  | otherwise = (1, w) : preFilter ws
  where alterFirst f [] = []
        alterFirst f (x:xs) = (f x) : xs

-- Unify the number of occurences
bag :: Num a => [(a, T.Text)] -> [(a, T.Text)]
bag = map (sum . map fst &&& snd . head) . groupBy (curry $ uncurry (==) . both snd) . sortBy (comparing snd)

-- The stemming method from NLP.Stemmer, wrapped up to use Text insead of String
stem' :: Stemmer -> T.Text -> T.Text
stem' lang = T.pack . stem lang . T.unpack

-- The set of words in a dataset that meet the common threshold
common :: [(a, T.Text)] -> S.Set T.Text
common dataset = S.fromList . map snd . filter (\w -> fst w * common_threshold >= datalen) $ (df (`S.notMember` stoplist) dataset)
  where datalen = length dataset

-- Number of documents with each word not excluded by filter
df :: (T.Text -> Bool) -> [(a, T.Text)] -> [(Int, T.Text)]
df' :: (T.Text -> Bool) -> [(a, T.Text)] -> [(Float, T.Text)]
df' wf = map (log . balance . sum . map fst &&& snd . head) . groupBy ((==) `on` snd) . sortBy (comparing snd) . concatMap (bag . filter (wf . snd) . preFilter . words' . snd)
df wf = map (length &&& head) . group . sort . concatMap (map head . group . sort . filter wf . map snd . preFilter . words' . snd)
--df wf = map (length &&& head) . group . sort . concatMap (map head . group . sort . filter wf . words' . snd)

-- Calculates the Delta Inverse Document Frequency of the dataset
didf :: (Num a, Ord a) => S.Set T.Text -> [(a, T.Text)] -> M.Map T.Text Float
didf legalWords dataset = uncurry (M.unionWith (+)) . both (M.fromList . map swap) .
    (                     df' (`S.member` legalWords) . filter ((>0) . fst) &&&
     map (first negate) . df' (`S.member` legalWords) . filter ((<0) . fst)) $ dataset


massCorr :: Floating a => S.Set T.Text -> [(a, T.Text)] -> M.Map T.Text a
massCorr legalWords dataset = let (n, ex, ex2, eys) = moments in
    M.map (\(ey, ey2, exy) -> (exy - ex*ey/n) / sqrt ((ex2 - ex^2/n) * (ey2 - ey^2/n))) eys
  where initial = (0, 0, 0,                                                   -- n, E[x], E[x^2]
                   M.fromList (zip (S.toList legalWords) $ repeat (0, 0, 0))) -- E[y], E[y^2], E[xy]
        --updateInner (n, ex, ex2, eys) (x, (y, w)) = (n, ex, ex2, M.adjust (\(ey, ey2, exy) -> (ey+y, ey2+y^2, exy+x*y)) w eys)
        --updateInner :: (Float, Float, Float, M.Map T.Text (Float, Float, Float)) -> (Float, (Float, T.Text)) -> (Float, Float, Float, M.Map T.Text (Float, Float, Float))
        updateInner (n, ex, ex2, eys) (x, (y, w)) = (n, ex, ex2, M.adjust (\(ey, ey2, exy) -> (ey+y, ey2+y^2, exy+x*y)) w eys)
        --updateOuter :: (Float, Float, Float, M.Map T.Text (Float, Float, Float)) -> (Float, a) -> (Float, Float, Float, M.Map T.Text (Float, Float, Float))
        updateOuter (n, ex, ex2, eys) (x, _) = (n+1, ex+x, ex2+x^2, eys) -- TODO update this with bag
        --tokens :: (Float, T.Text) -> [(Float, (Float, T.Text))]
        tokens str = let wordList = preFilter . words' $ snd str; n = fromIntegral (length wordList) in
        --tokens str = let n = fromIntegral . length . words' . snd $ str in
            zip (repeat $ fst str) (map (first (/n)) $ bag wordList)
            --sequence $ second (map ((/n) . fromIntegral . length &&& head) . group . sort . words') str
        moments = foldl' (\acc d -> foldl' updateInner (updateOuter acc d) (tokens d)) initial dataset

massCorr' :: [(Double, b)] -> LA.Matrix Double -> (LA.Vector Double, Double)
massCorr' ys xss = let cov = (exy - LA.scale ey ex)
                       sx = LA.cmap sqrt (ex2 - ex^2)
                       sy = LA.konst (sqrt $ ey2 - ey^2) w
                   in (sx/sy, LA.dot ex (sy/sx) - ey)
  where (n, w) = first fromIntegral $ LA.size xss
        initial = (0, 0, LA.konst 0 w, LA.konst 0 w, LA.konst 0 w)
        update (sy, sy2, sx, sx2, sxy) (y, xs) = (sy+y, sy2+y^2, sx+xs, sx2+xs^2, sxy + LA.scale y xs)
        (ey, ey2, ex, ex2, exy) = (sy/n, sy2/n, LA.scale (1/n) sx, LA.scale (1/n) sx2, LA.scale (1/n) sxy)
          where (sy, sy2, sx, sx2, sxy) = foldl' update initial $ zip (map fst ys) (LA.toRows xss)

--model :: M.Map T.Text Float -> T.Text -> Float
--model weights str = let tokens = preFilter (words' str) in
--  (sum . map (\(n, w) -> n * M.findWithDefault 0 w weights)) (bag tokens) / fromIntegral (length tokens)

t2v :: M.Map T.Text Int -> M.Map T.Text Double -> T.Text -> LA.Vector Double
t2v lut weights t = LA.fromList $ map (flip (IM.findWithDefault 0) (t2v' lut weights t)) [0 .. length lut - 1]

t2v' :: M.Map T.Text Int -> M.Map T.Text Double -> T.Text -> IM.IntMap Double
t2v' lut weights t = IM.fromList . map (\(x, w) -> (lut M.! w, realToFrac $ weights M.! w * x)) . filter ((`M.member` lut) . snd) . bag . preFilter $ wordList
  where wordList = words' t
        n = fromIntegral $ length wordList
-- x, can also be x * weights!w or x * weights!w / n

data2m :: M.Map T.Text Int -> M.Map T.Text Double -> [(a, T.Text)] -> LA.Matrix Double
data2m lut weights = LA.fromRows . map (t2v lut weights . snd)

m2p :: [(Double, a)] -> LA.Matrix Double -> [(Double, IM.IntMap Double)]
m2p dataset = zip (map (clamp . fst) dataset) . map (\row -> IM.fromList $ map (id &&& (row LA.!)) [0 .. LA.size row - 1]) . LA.toRows

checkProblem :: Model -> [(Double, IM.IntMap Double)] -> IO [(Double, Double)]
checkProblem model = sequence . map (sequence . second (predict model))
--main = do
--  dataset <- trainingSamples
--  putStrLn "Data Loaded"
--  let legalWords = common dataset
--  let weights = didf legalWords dataset
--  --let weights = massCorr legalWords $ map (\(x, t) -> (if x > 0 then 1 else -1, t)) dataset
--  putStrLn "word scores:"
--  putStrLn . show $ sortBy (comparing snd) (M.toList weights)
--  let results = map (second $ model weights) dataset
--  let adjustment = r2 results
--  let results' =  map (\(x, t) -> ((x, crop $ model weights t * fst (snd adjustment) + snd (snd adjustment)), t)) dataset
--  let errors = sortBy (comparing $ abs . uncurry (-) . fst) results'
--  let accuracy = fromIntegral (length $ filter (uncurry (==) . both (>0) . fst) results') / fromIntegral (length results')
--  putStrLn $ "r²=" ++ show (fst adjustment) ++ "\taccuracy=" ++ show accuracy
--  putStrLn "least accurate:"
--  putStrLn . show $ take 50 (reverse errors)

main = do
  dataset <- trainingSamples
  let legalWords = common dataset
  let weights = realToFrac <$> didf legalWords dataset
  --let wordIndexes = let list = S.toList legalWords -- map fst . sortBy (comparing snd) . M.toList $ weights
  --                  in  M.fromList . (flip zip) [0..] $ take 300 list ++ take 300 (reverse list)
  putStrLn $ "words: (" ++ show (length legalWords) ++ ")"
  --putStrLn . show $ S.toList legalWords
  let wordIndexes = M.fromList $ zip (S.toList legalWords) [0..]
  let (u, s, v) = LA.thinSVD (data2m wordIndexes weights dataset)
  let nsv = 300
  let (u', s', v') = (u LA.?? (LA.All, LA.Take nsv), LA.subVector 0 nsv s, v LA.?? (LA.All, LA.Take nsv))
  --let (weights2, bias) = massCorr' dataset u'
  --putStrLn . show $ (weights2, bias)
  let dataset' = m2p dataset (u' LA.<> LA.diag s')
  svmModel <- train (CSvc 1.5) (RBF 0.01) dataset'
  resultsTraining <- checkProblem svmModel dataset'
  --let resultsTraining = zip (map fst dataset) (LA.toList $ (u' LA.#> weights2) + LA.konst bias nsv)
  validationDataset <- validationSamples
  --let u'2 = (LA.<> v') . data2m wordIndexes weights $ validationDataset
  --let resultsValidation = zip (map fst validationDataset) (LA.toList $ (u'2 LA.#> weights2) + LA.konst bias nsv)
  resultsValidation <- checkProblem svmModel . m2p validationDataset . (LA.<> v') . data2m wordIndexes weights $ validationDataset
  --let dataset' = map (\(x, t) -> (clamp x, t2v' wordIndexes t)) dataset
  --svmModel <- train (CSvc 10) Linear (dropEvery 5 dataset')
  --resultsTraining <- sequence $ map (sequence . second (predict svmModel)) (dropEvery 5 $ dataset')
  --resultsValidation <- sequence $ map (sequence . second (predict svmModel)) (takeEvery 5 $ dataset')
  let accuracy results = let a = (length $ filter (\(a, b) -> (a>0) == (b>0)) results)
                             n = (length results)
                         in show a ++ "/" ++ show n ++ " (" ++ show (100 * fromIntegral a / fromIntegral n) ++ "%)" -- corr=" ++ show (r2 results)
  putStrLn $ "training accuracy: " ++ accuracy resultsTraining ++ "\tvalidation accuracy: " ++ accuracy resultsValidation

crop :: (Num a, Ord a) => a -> a
crop = max (-1) . min 1

clamp :: (Num a, Ord a) => a -> a
clamp x = if x > 0 then 1 else -1

balance :: (Ord a, Fractional a) => a -> a
balance x = if x > 0 then 1 + x else 1 / (1 - x)

takeEvery :: Int -> [a] -> [a]
takeEvery n xs = case drop (n-1) xs of (y:ys) -> y : takeEvery n ys; [] -> []

dropEvery :: Int -> [a] -> [a]
dropEvery n xs = case splitAt (n-1) xs of (x, (y:ys)) -> x ++ dropEvery n ys; (x, []) -> x

r2 :: Floating a => [(a, a)] -> (a, (a, a))
r2 points = let n = fromIntegral (length points)
                ex = sum (map fst points) / n
                ey = sum (map snd points) / n
                sx = sqrt $ sum (map ((^2) . ((-)ex) . fst) points) / n
                sy = sqrt $ sum (map ((^2) . ((-)ey) . snd) points) / n
                cov = sum (map (\(x, y) -> (x - ex) * (y - ey)) points) / n
            in (cov / (sx * sy), (sx / sy, ex * sy / sx - ey))

both :: (a -> b) -> (a, a) -> (b, b)
both f (x, y) = (f x, f y)

swap :: (a, b) -> (b, a)
swap (x, y) = (y, x)
