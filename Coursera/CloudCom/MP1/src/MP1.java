import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.nio.file.Files;
import java.nio.file.Paths;
//import java.lang.reflect.Array;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.*;

public class MP1 {
    Random generator;
    String userName;
    String inputFileName;
    String delimiters = " \t,;.?!-:@[](){}_*/";
    String[] stopWordsArray = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
            "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its",
            "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
            "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
            "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
            "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
            "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
            "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
            "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
            "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"};

    void initialRandomGenerator(String seed) throws NoSuchAlgorithmException {
        MessageDigest messageDigest = MessageDigest.getInstance("SHA");
        messageDigest.update(seed.toLowerCase().trim().getBytes());
        byte[] seedMD5 = messageDigest.digest();

        long longSeed = 0;
        for (int i = 0; i < seedMD5.length; i++) {
            longSeed += ((long) seedMD5[i] & 0xffL) << (8 * i);
        }

        this.generator = new Random(longSeed);
    }

    Integer[] getIndexes() throws NoSuchAlgorithmException {
        Integer n = 10000;
        Integer number_of_lines = 50000;
//    	Integer n = 5;
//    	Integer number_of_lines = 20;
        Integer[] ret = new Integer[n];
        this.initialRandomGenerator(this.userName);
        for (int i = 0; i < n; i++) {
            ret[i] = generator.nextInt(number_of_lines);
        }
        return ret;
    }

    public MP1(String userName, String inputFileName) {
        this.userName = userName;
        this.inputFileName = inputFileName;
    }

    class ValueComparator implements Comparator<String> {
    	 
        Map<String, Integer> map;
     
        public ValueComparator(Map<String, Integer> base) {
            this.map = base;
        }
     
        public int compare(String a, String b) {
	            if (map.get(a) > map.get(b)) return -1;
	            else if(map.get(a) == map.get(b)){
	            	return a.compareTo(b);
	            }
	            else return 1;
	            // returning 0 would merge keys 
        }
    }
    
    public String[] process() throws Exception {
        String[] ret = new String[20];
       
        //TODO
        
		Integer[] indices = getIndexes();
		List<Integer> il = new ArrayList<Integer>(Arrays.asList(indices));
    	System.out.println(il);
		

    	File f = new File(inputFileName);
        String s = null;
        
        List<String> swl = new ArrayList<String>(Arrays.asList(stopWordsArray));
        Map<String, Integer> wm = new HashMap<String, Integer>();       
        
        for (Integer x: il){
        	
            FileReader fr = new FileReader(f);
            BufferedReader br = new BufferedReader(fr);
            
        	for(int ln = 0; ln <= x; ln++) s= br.readLine();	
            
            
        	s= s.toLowerCase();
        	s= s.trim();
        	System.out.println("Line # is:: " + x + "  String is::" + s);
        	
//        	System.out.println("Line is::"+ s);
        	StringTokenizer st = new StringTokenizer(s,delimiters);
        	while (st.hasMoreTokens()){
        		String tempToken = st.nextToken();
        		if (swl.contains(tempToken) != true){
        			if(wm.containsKey(tempToken)) wm.put(tempToken, wm.get(tempToken)+1);
        			else wm.put(tempToken, 1);
//        			System.out.print(" " + tempToken);
        		} 
        	}
        }

        ValueComparator vc =  new ValueComparator(wm);
		TreeMap<String,Integer> sortedMap = new TreeMap<String,Integer>(vc);
		sortedMap.putAll(wm);
//		System.out.println(sortedMap);
		
		//		String fKey = sortedMap.firstKey();
//		System.out.println(fKey);

//		System.out.println("Tree Map Size:: " + sortedMap.entrySet().size());
		
		int i = 0;
		String toKey = null;
//		System.out.println(sortedMap.keySet());
		
		for (String key : sortedMap.keySet()) {
		    if (i == 20) {
		        toKey = key;
		        break;
		    }
		    ret[i] = key;
		    i++;
		}
		
//		System.out.println("-------"+ toKey);
		
		SortedMap<String,Integer> t20Map = sortedMap.headMap(toKey);
//		
//		i = 0;
		System.out.println(t20Map.entrySet());
//		for (String x:t20Map.keySet()){
//			System.out.println(x + " --" + i);
////			ret[i] = x;
//			i++;
//		}
		
//		sortedMap.ent
        return ret;
    }

    public static void main(String[] args) throws Exception {
        if (args.length < 1){
            System.out.println("MP1 <User ID>");
        }
        else {
            String userName = args[0];
            String inputFileName = "./input.txt";
            MP1 mp = new MP1(userName, inputFileName);
            String[] topItems = mp.process();
            for (String item: topItems){
                System.out.println(item);
            }
        }
    }
}
